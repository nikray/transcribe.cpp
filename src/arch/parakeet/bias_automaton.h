// arch/parakeet/bias_automaton.h - custom-vocabulary biasing (word
// boosting / shallow fusion) for the Parakeet greedy decoders.
//
// Parakeet has no word vocabulary to extend — words are spelled from a
// fixed set of ~1k SentencePiece subword pieces. "Custom vocabulary"
// therefore means boosting, at decode time, the scores of the token
// sequences that spell the user's words, so near-miss recognitions flip
// toward the intended spelling.
//
// Design follows NVIDIA's GPU-PB / "TurboBias" phrase boosting
// (arXiv:2508.07014, the greedy-compatible mechanism NeMo ships for
// pure TDT models), reduced to what per-step greedy decoding can
// express:
//
//   * Words are segmented into token-id sequences against the model's
//     piece vocabulary (with case variants, honoring the SentencePiece
//     "▁" word-start marker).
//   * The sequences are compiled into an Aho-Corasick automaton (prefix
//     trie + failure links). During decoding the automaton state is
//     tracked over the emitted tokens; failure links make a dead
//     partial match fall back to the longest suffix that is still a
//     live prefix — the greedy analogue of TurboBias's boost
//     cancellation (there is no accumulated path score to retract in
//     greedy, so cancellation degenerates to correct state fallback).
//   * At each decode step, an additive bias is applied to the token
//     logits BEFORE argmax: continuations of the current partial match
//     and word-start tokens get `boost` logit units; an arc that
//     completes a word gets an extra completion bonus. Blank and (for
//     TDT) the duration-head logits are never touched, and reported
//     token confidences stay derived from the UNBIASED logits.
//
// This module is pure host C++ (no ggml, no tokenizer dependency) so it
// is unit-testable in isolation; the decode loops in decoder.cpp take a
// `const BiasAutomaton *` and a caller-owned state integer. Everything
// here is const-after-build and safe to share across threads.
//
// Known limitations (accepted for v1, mirroring the reference
// implementation this was ported from):
//   * Greedy boosting flips near-miss decisions only; confidently-wrong
//     tokens stay wrong.
//   * Case-variant generation folds ASCII letters only; a non-ASCII
//     word is boosted in its given spelling (further variants are
//     simply not generated).

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace transcribe::parakeet {

// Additive logit boost per matched token when the caller does not pick
// one. Tuned empirically on parakeet-tdt-0.6b: ~1.8 flips nothing; 3.0
// rescues near-miss invented nouns with no damage to phonetic
// neighbors; 10.0 visibly over-boosts (merged/hallucinated words).
inline constexpr float BIAS_DEFAULT_BOOST = 3.0f;

// Bonus factor added on top of `boost` for an arc that completes a
// whole word — rewards committing the final piece of "ChargeBee" over a
// lexically similar non-word continuation.
inline constexpr float BIAS_COMPLETION_BONUS_FACTOR = 0.5f;

// ---------------------------------------------------------------------------
// Segmentation (pure)
// ---------------------------------------------------------------------------

// Case variants worth boosting. The piece vocabulary is cased, and STT
// output may render the word at a sentence start or lowercased
// mid-flow. Returns {word, lower(word), Capitalized(lower(word))} with
// duplicates removed, preserving that order; an all-whitespace word
// yields an empty list. ASCII-only case folding (see header comment).
std::vector<std::string> bias_word_variants(const std::string & word);

// piece string -> token id with the metadata segmentation needs.
// First occurrence wins on duplicate pieces (duplicates shouldn't
// exist in a SentencePiece vocab). max_piece_len is in BYTES.
struct BiasPieceIndex {
    std::unordered_map<std::string, int> piece_to_id;
    size_t                               max_piece_len = 0;
};

BiasPieceIndex build_bias_piece_index(const std::vector<std::string> & pieces);

// Greedy longest-match segmentation of one word (single variant) into
// token ids, as it would appear at a word start ("▁" prefix).
//
// The runtime has the id→piece table but not the SentencePiece
// encoder, so segmentation matches piece strings directly. Greedy
// longest match is not guaranteed to reproduce SentencePiece's Viterbi
// segmentation, but for boosting purposes any valid spelling of the
// word is a usable path. Matching runs on bytes; because every piece
// is a valid UTF-8 substring, byte-longest and codepoint-longest
// prefixes coincide. Returns an empty vector when some position has no
// matching piece (word not spellable — e.g. characters outside the
// model's alphabet).
std::vector<int> bias_segment_word(const std::string & word, const BiasPieceIndex & index);

// Segment all case variants of all words against `pieces` (the model's
// id→piece table, index == token id). Identical variant sequences are
// deduplicated; a word none of whose variants segment lands in
// `failed`.
struct BiasSegmentation {
    std::vector<std::vector<int>> seqs;
    std::vector<std::string>      failed;
};

BiasSegmentation bias_segment_vocabulary(const std::vector<std::string> & words,
                                         const std::vector<std::string> & pieces);

// ---------------------------------------------------------------------------
// Aho-Corasick automaton with precomputed dense bias rows
// ---------------------------------------------------------------------------

// States are dense indices; 0 is the root. The per-state additive bias
// row over the token classes is precomputed at build time (const after
// build, thread-safe to share). Row semantics: continuations of the
// live partial match and fresh word starts both get `boost`; an arc
// that completes a word gets the completion bonus on top; when a token
// continues the match AND starts a new word the larger value wins.
// Blank (index width-1) and out-of-range ids are never boosted.
class BiasAutomaton {
  public:
    BiasAutomaton() = default;

    // Compile `seqs` (from bias_segment_vocabulary) into an automaton
    // whose rows are `width` floats wide. `width` is the token-class
    // count INCLUDING blank at width-1 (== pred_vocab for TDT/RNNT,
    // n_classes for CTC). boost <= 0 or width <= 0 or empty seqs yield
    // an empty() automaton that callers should not consult.
    static BiasAutomaton build(const std::vector<std::vector<int>> & seqs, int width, float boost);

    bool empty() const { return nodes_.size() <= 1 || width_ <= 0; }

    int width() const { return width_; }

    float boost() const { return boost_; }

    int n_nodes() const { return static_cast<int>(nodes_.size()); }

    int root() const { return 0; }

    // Aho-Corasick transition on one emitted token. Falls back through
    // failure links until a goto arc matches (or lands on root).
    // `state` outside [0, n_nodes) is treated as root (defensive).
    int advance(int state, int token_id) const;

    // Dense additive bias row for `state` (width() floats). `state`
    // outside [0, n_nodes) returns the root row. Never null on a
    // non-empty automaton.
    const float * row(int state) const;

    // Introspection for tests.
    bool terminal(int state) const;
    int  depth(int state) const;

  private:
    struct Node {
        std::unordered_map<int, int> children;  // token id -> node index
        int                          fail     = 0;
        int                          depth    = 0;
        bool                         terminal = false;
    };

    std::vector<Node>               nodes_;
    std::vector<std::vector<float>> rows_;  // one dense row per node
    int                             width_ = 0;
    float                           boost_ = 0.0f;
};

}  // namespace transcribe::parakeet
