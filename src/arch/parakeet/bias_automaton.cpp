// arch/parakeet/bias_automaton.cpp - see bias_automaton.h.

#include "bias_automaton.h"

#include <algorithm>
#include <deque>
#include <set>

namespace transcribe::parakeet {

namespace {

// SentencePiece word-start marker U+2581 ("▁", UTF-8 E2 96 81).
constexpr const char k_word_start[] = "\xE2\x96\x81";

std::string ascii_lower(const std::string & s) {
    std::string out = s;
    for (char & c : out) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return out;
}

std::string strip_ws(const std::string & s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return std::string();
    }
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

}  // namespace

// ---------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------

std::vector<std::string> bias_word_variants(const std::string & word_in) {
    const std::string word = strip_ws(word_in);
    if (word.empty()) {
        return {};
    }
    std::vector<std::string> variants;
    variants.push_back(word);

    const std::string lower = ascii_lower(word);
    if (std::find(variants.begin(), variants.end(), lower) == variants.end()) {
        variants.push_back(lower);
    }
    std::string cap = lower;
    if (!cap.empty() && cap[0] >= 'a' && cap[0] <= 'z') {
        cap[0] = static_cast<char>(cap[0] - 'a' + 'A');
    }
    if (std::find(variants.begin(), variants.end(), cap) == variants.end()) {
        variants.push_back(cap);
    }
    return variants;
}

BiasPieceIndex build_bias_piece_index(const std::vector<std::string> & pieces) {
    BiasPieceIndex idx;
    idx.piece_to_id.reserve(pieces.size());
    for (size_t i = 0; i < pieces.size(); ++i) {
        const std::string & p = pieces[i];
        idx.piece_to_id.emplace(p, static_cast<int>(i));  // first occurrence wins
        idx.max_piece_len = std::max(idx.max_piece_len, p.size());
    }
    return idx;
}

std::vector<int> bias_segment_word(const std::string & word, const BiasPieceIndex & index) {
    const std::string target = std::string(k_word_start) + word;

    std::vector<int> ids;
    size_t           pos = 0;
    while (pos < target.size()) {
        int          match_id = -1;
        const size_t max_take = std::min(index.max_piece_len, target.size() - pos);
        // Longest piece first.
        for (size_t take = max_take; take >= 1; --take) {
            const auto it = index.piece_to_id.find(target.substr(pos, take));
            if (it != index.piece_to_id.end()) {
                match_id = it->second;
                pos += take;
                break;
            }
        }
        if (match_id < 0) {
            return {};
        }
        ids.push_back(match_id);
    }
    return ids;
}

BiasSegmentation bias_segment_vocabulary(const std::vector<std::string> & words,
                                         const std::vector<std::string> & pieces) {
    const BiasPieceIndex idx = build_bias_piece_index(pieces);

    BiasSegmentation           out;
    std::set<std::vector<int>> seen;
    for (const std::string & word : words) {
        bool any_variant = false;
        for (const std::string & variant : bias_word_variants(word)) {
            std::vector<int> ids = bias_segment_word(variant, idx);
            if (!ids.empty()) {
                if (seen.insert(ids).second) {
                    out.seqs.push_back(std::move(ids));
                }
                any_variant = true;
            }
        }
        if (!any_variant && !strip_ws(word).empty()) {
            out.failed.push_back(word);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Automaton
// ---------------------------------------------------------------------------

BiasAutomaton BiasAutomaton::build(const std::vector<std::vector<int>> & seqs, int width, float boost) {
    BiasAutomaton a;
    a.width_ = width;
    a.boost_ = boost;
    a.nodes_.emplace_back();  // root at index 0

    if (width <= 0) {
        return a;
    }

    // Trie.
    for (const std::vector<int> & seq : seqs) {
        if (seq.empty()) {
            continue;
        }
        int node = 0;
        for (int tid : seq) {
            auto it = a.nodes_[static_cast<size_t>(node)].children.find(tid);
            if (it == a.nodes_[static_cast<size_t>(node)].children.end()) {
                const int child = static_cast<int>(a.nodes_.size());
                a.nodes_.emplace_back();
                a.nodes_.back().depth                                 = a.nodes_[static_cast<size_t>(node)].depth + 1;
                a.nodes_[static_cast<size_t>(node)].children[tid]     = child;
                node                                                  = child;
            } else {
                node = it->second;
            }
        }
        a.nodes_[static_cast<size_t>(node)].terminal = true;
    }

    // BFS failure links (classic Aho-Corasick). Root's children fail to
    // root; deeper children walk the parent's failure chain.
    std::deque<int> queue;
    for (const auto & kv : a.nodes_[0].children) {
        a.nodes_[static_cast<size_t>(kv.second)].fail = 0;
        queue.push_back(kv.second);
    }
    while (!queue.empty()) {
        const int node = queue.front();
        queue.pop_front();
        for (const auto & kv : a.nodes_[static_cast<size_t>(node)].children) {
            const int tid   = kv.first;
            const int child = kv.second;
            int       f     = a.nodes_[static_cast<size_t>(node)].fail;
            while (f != 0 && a.nodes_[static_cast<size_t>(f)].children.count(tid) == 0) {
                f = a.nodes_[static_cast<size_t>(f)].fail;
            }
            const auto fit  = a.nodes_[static_cast<size_t>(f)].children.find(tid);
            int        dest = (fit != a.nodes_[static_cast<size_t>(f)].children.end()) ? fit->second : 0;
            if (dest == child) {  // defensive: a state must not fail to itself
                dest = 0;
            }
            a.nodes_[static_cast<size_t>(child)].fail = dest;
            queue.push_back(child);
        }
    }

    // Dense per-state bias rows, precomputed so decode-time lookups are
    // a plain pointer read (and the automaton stays const / sharable).
    const float completion = boost * (1.0f + BIAS_COMPLETION_BONUS_FACTOR);
    a.rows_.assign(a.nodes_.size(), std::vector<float>(static_cast<size_t>(width), 0.0f));
    for (size_t s = 0; s < a.nodes_.size(); ++s) {
        std::vector<float> & row = a.rows_[s];
        auto                 add = [&](const Node & n) {
            for (const auto & kv : n.children) {
                const int tid = kv.first;
                if (tid < 0 || tid >= width - 1) {  // exclude blank at width-1 and OOR ids
                    continue;
                }
                const float value = a.nodes_[static_cast<size_t>(kv.second)].terminal ? completion : boost;
                row[static_cast<size_t>(tid)] = std::max(row[static_cast<size_t>(tid)], value);
            }
        };
        add(a.nodes_[s]);
        if (s != 0) {
            add(a.nodes_[0]);  // fresh word starts stay live mid-match
        }
    }

    return a;
}

int BiasAutomaton::advance(int state, int token_id) const {
    if (state < 0 || state >= static_cast<int>(nodes_.size())) {
        state = 0;
    }
    int node = state;
    while (true) {
        const Node & n  = nodes_[static_cast<size_t>(node)];
        const auto   it = n.children.find(token_id);
        if (it != n.children.end()) {
            return it->second;
        }
        if (node == 0) {
            return 0;
        }
        node = n.fail;
    }
}

const float * BiasAutomaton::row(int state) const {
    if (state < 0 || state >= static_cast<int>(rows_.size())) {
        state = 0;
    }
    return rows_[static_cast<size_t>(state)].data();
}

bool BiasAutomaton::terminal(int state) const {
    if (state < 0 || state >= static_cast<int>(nodes_.size())) {
        return false;
    }
    return nodes_[static_cast<size_t>(state)].terminal;
}

int BiasAutomaton::depth(int state) const {
    if (state < 0 || state >= static_cast<int>(nodes_.size())) {
        return 0;
    }
    return nodes_[static_cast<size_t>(state)].depth;
}

}  // namespace transcribe::parakeet
