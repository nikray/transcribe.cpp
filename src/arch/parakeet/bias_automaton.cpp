// arch/parakeet/bias_automaton.cpp - see bias_automaton.h.

#include "bias_automaton.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <set>
#include <stdexcept>

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

bool ascii_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

}  // namespace

std::string bias_normalize_term(const std::string & term) {
    std::string out;
    bool        space = false;
    for (unsigned char c : term) {
        if (ascii_space(c)) {
            space = !out.empty();
        } else {
            if (space) {
                out += ' ';
            }
            out += static_cast<char>(c);
            space = false;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------

std::vector<std::string> bias_word_variants(const std::string & word_in) {
    const std::string word = bias_normalize_term(word_in);
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
    // SentencePiece uses its word-start marker at every whitespace boundary.
    // Collapse ASCII whitespace and trim, matching the model's text projection.
    std::string target;
    bool        boundary = true;
    for (unsigned char ch : word) {
        if (ascii_space(ch)) {
            boundary = true;
            continue;
        }
        if (boundary) {
            target += k_word_start;
            boundary = false;
        }
        target += static_cast<char>(ch);
    }
    if (target.empty()) {
        return {};
    }
    // Prefer the longest piece that still permits a complete spelling.
    std::vector<size_t> next(target.size() + 1, 0);
    std::vector<int>    chosen(target.size(), -1);
    next[target.size()] = target.size();
    for (size_t pos = target.size(); pos-- > 0;) {
        const size_t max_take = std::min(index.max_piece_len, target.size() - pos);
        for (size_t take = max_take; take > 0; --take) {
            if (next[pos + take] == 0) {
                continue;
            }
            const auto it = index.piece_to_id.find(target.substr(pos, take));
            if (it != index.piece_to_id.end()) {
                next[pos]   = pos + take;
                chosen[pos] = it->second;
                break;
            }
        }
    }
    if (next[0] == 0) {
        return {};
    }
    std::vector<int> ids;
    for (size_t pos = 0; pos < target.size(); pos = next[pos]) {
        ids.push_back(chosen[pos]);
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
                // A first piece that is just the word-start marker (or
                // marker + one char) is a GENERIC start — flag it so the
                // automaton gives its root arc no standing boost (see
                // BiasSegmentation::generic_start_ids).
                const int           first        = ids.front();
                const std::string & fp           = pieces[static_cast<size_t>(first)];
                const size_t        marker_bytes = sizeof(k_word_start) - 1;
                if (fp.compare(0, marker_bytes, k_word_start) == 0) {
                    const size_t suffix_bytes = fp.size() - marker_bytes;
                    const auto   first_byte   = suffix_bytes > 0 ? static_cast<unsigned char>(fp[marker_bytes]) : 0;
                    const size_t char_bytes = first_byte < 0x80 ? 1 : first_byte < 0xE0 ? 2 : first_byte < 0xF0 ? 3 : 4;
                    if (suffix_bytes == 0 || suffix_bytes == char_bytes) {
                        out.generic_start_ids.insert(first);
                    }
                }
                if (seen.insert(ids).second) {
                    out.seqs.push_back(std::move(ids));
                }
                any_variant = true;
            }
        }
        if (!any_variant && !bias_normalize_term(word).empty()) {
            out.failed.push_back(word);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Automaton
// ---------------------------------------------------------------------------

std::vector<BiasAutomaton::Node> BiasAutomaton::build_trie(const std::vector<std::vector<int>> & seqs,
                                                           int                                   width,
                                                           size_t                                max_dense_bytes) {
    std::vector<Node> nodes(1);
    if (width <= 0) {
        return nodes;
    }
    const size_t max_nodes = max_dense_bytes / (static_cast<size_t>(width) * sizeof(float));
    if (max_nodes == 0) {
        throw std::length_error("Parakeet vocabulary exceeds dense-row memory budget");
    }
    // Trie.
    for (const std::vector<int> & seq : seqs) {
        if (seq.empty()) {
            continue;
        }
        int node = 0;
        for (int tid : seq) {
            auto it = nodes[static_cast<size_t>(node)].children.find(tid);
            if (it == nodes[static_cast<size_t>(node)].children.end()) {
                if (nodes.size() >= max_nodes) {
                    throw std::length_error("Parakeet vocabulary exceeds dense-row memory budget");
                }
                const int child = static_cast<int>(nodes.size());
                nodes.emplace_back();
                nodes.back().depth                             = nodes[static_cast<size_t>(node)].depth + 1;
                nodes[static_cast<size_t>(node)].children[tid] = child;
                node                                           = child;
            } else {
                node = it->second;
            }
        }
        nodes[static_cast<size_t>(node)].terminal = true;
    }

    return nodes;
}

void BiasAutomaton::validate_size(const std::vector<std::vector<int>> & seqs, int width, size_t max_dense_bytes) {
    (void) build_trie(seqs, width, max_dense_bytes);
}

BiasAutomaton BiasAutomaton::build(const std::vector<std::vector<int>> & seqs,
                                   int                                   width,
                                   float                                 boost,
                                   const std::unordered_set<int> &       generic_start_ids,
                                   size_t                                max_dense_bytes) {
    BiasAutomaton a;
    a.width_ = width;
    a.boost_ = boost;
    a.nodes_.emplace_back();  // root at index 0

    if (!std::isfinite(boost) || boost < 0.0f || boost > BIAS_MAX_BOOST) {
        throw std::invalid_argument("Parakeet vocabulary boost must be finite and between 0 and 10");
    }
    if (width <= 0 || boost == 0.0f) {
        return a;
    }

    a.nodes_ = build_trie(seqs, width, max_dense_bytes);

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
    // Arc base depends on the arc's SOURCE: continuations (from a live
    // partial match) get full `boost`; fresh word starts (from the root)
    // get the down-weighted base (see BIAS_WORD_START_FACTOR); an arc
    // that completes a word gets the completion bonus on its base.
    const float word_start_base = boost * BIAS_WORD_START_FACTOR;
    a.rows_.assign(a.nodes_.size(), std::vector<float>(static_cast<size_t>(width), 0.0f));
    for (size_t s = 0; s < a.nodes_.size(); ++s) {
        std::vector<float> & row = a.rows_[s];
        auto                 add = [&](const Node & n, float base) {
            for (const auto & kv : n.children) {
                const int tid = kv.first;
                if (tid < 0 || tid >= width - 1) {  // exclude blank at width-1 and OOR ids
                    continue;
                }
                const float value             = a.nodes_[static_cast<size_t>(kv.second)].terminal ?
                                                    base * (1.0f + BIAS_COMPLETION_BONUS_FACTOR) :
                                                    base;
                row[static_cast<size_t>(tid)] = std::max(row[static_cast<size_t>(tid)], value);
            }
        };
        // Every failure-chain state is a live suffix of the emitted tokens.
        // Include its continuations as well as the longest current match.
        for (int state = static_cast<int>(s); state != 0; state = a.nodes_[state].fail) {
            add(a.nodes_[state], boost);
        }
        add(a.nodes_[0], word_start_base);
        // Generic word-start arcs (bare marker / marker+1 char pieces) get
        // NO standing boost: they are too unspecific to be a useful entry
        // signal and are exactly the arcs that steal real speech. Zero the
        // row slots for root arcs whose token is generic — unless the same
        // token is also a live continuation from this state, which keeps
        // its (larger) continuation value when any live suffix contains it.
        for (const int tid : generic_start_ids) {
            if (tid < 0 || tid >= width - 1) {
                continue;
            }
            const bool is_root_arc = a.nodes_[0].children.count(tid) != 0;
            bool       is_own_arc  = false;
            for (int state = static_cast<int>(s); state != 0; state = a.nodes_[state].fail) {
                is_own_arc = is_own_arc || a.nodes_[state].children.count(tid) != 0;
            }
            if (is_root_arc && !is_own_arc) {
                row[static_cast<size_t>(tid)] = 0.0f;
            }
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
