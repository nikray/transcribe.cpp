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
//     get `boost` logit units, fresh word-start tokens get
//     `boost * BIAS_WORD_START_FACTOR` (down-weighted — the standing
//     word-start mass scales with vocabulary size; see the constant),
//     and an arc that completes a word gets an extra completion bonus
//     on its base. Blank and (for TDT) the duration-head logits are
//     never touched, and reported token confidences stay derived from
//     the UNBIASED logits.
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

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace transcribe::parakeet {

// Additive logit boost per matched token when the caller does not pick
// one. Tuned empirically on parakeet-tdt-0.6b: ~1.8 flips nothing; 3.0
// rescues near-miss invented nouns with no damage to phonetic
// neighbors; 10.0 visibly over-boosts (merged/hallucinated words).
inline constexpr float  BIAS_DEFAULT_BOOST   = 3.0f;
inline constexpr float  BIAS_MAX_BOOST       = 10.0f;
inline constexpr size_t BIAS_MAX_TERMS       = 256;
inline constexpr size_t BIAS_MAX_TERM_BYTES  = 256;
inline constexpr size_t BIAS_MAX_DENSE_BYTES = 64 * 1024 * 1024;

// Bonus factor added on top of an arc's base boost for an arc that
// completes a whole word — rewards committing the final piece of
// "ChargeBee" over a lexically similar non-word continuation.
inline constexpr float BIAS_COMPLETION_BONUS_FACTOR = 0.5f;

// Weight applied to fresh word-START arcs (arcs leaving the automaton
// root), relative to continuation arcs (arcs leaving a live partial
// match). Word starts are pure prior — every vocab word contributes one
// standing boosted arc at every decode step — so their total mass
// scales linearly with the vocabulary and, at full weight, starts
// winning the argmax on flat logits (silence/noise) once the list
// reaches a few dozen words: a 50-word list at boost 3.0 was measured
// hallucinating boosted words on 9/37 silence-ish corpus clips.
// Continuations carry acoustic evidence (the match already began) and
// keep full boost — near-miss flips like "Quelium"→"Quellium" are
// decided at continuation tokens and are unaffected.
inline constexpr float BIAS_WORD_START_FACTOR = 0.5f;

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

// Longest viable piece segmentation of one word/phrase (single variant) into
// token ids, as it would appear at a word start ("▁" prefix).
//
// Match the loaded SentencePiece-style pieces, inserting the word-start
// marker at every ASCII whitespace boundary. A backward reachability pass
// finds a complete spelling without greedy dead ends. This intentionally
// does not claim canonical SentencePiece encoding: existing Parakeet GGUFs
// do not reliably preserve the source tokenizer algorithm/normalizer.
// Returns empty if no complete spelling exists.
std::vector<int> bias_segment_word(const std::string & word, const BiasPieceIndex & index);

// Segment all case variants of all words against `pieces` (the model's
// id→piece table, index == token id). Identical variant sequences are
// deduplicated; a word none of whose variants segment lands in
// `failed`.
struct BiasSegmentation {
    std::vector<std::vector<int>> seqs;
    std::vector<std::string>      failed;
    // Token ids whose piece is a GENERIC word start — the bare "\xE2\x96\x81"
    // marker or marker + a single character — collected from the pieces
    // that appear as the FIRST token of some sequence. Such arcs are too
    // unspecific to be a useful entry signal (a standing boost on "\xE2\x96\x81C"
    // steals real speech: "say" -> "C" + natural continuation -> "C."),
    // so BiasAutomaton::build gives their root arcs no standing boost.
    // A match can still begin through them — the automaton advances on
    // whatever the decoder emits — and continuations keep full boost.
    std::unordered_set<int>       generic_start_ids;
};

BiasSegmentation bias_segment_vocabulary(const std::vector<std::string> & words,
                                         const std::vector<std::string> & pieces);

// ---------------------------------------------------------------------------
// Aho-Corasick automaton with precomputed dense bias rows
// ---------------------------------------------------------------------------

// States are dense indices; 0 is the root. The per-state additive bias
// row over the token classes is precomputed at build time (const after
// build, thread-safe to share). Row semantics: continuations of the
// live partial match get `boost`; fresh word starts (arcs leaving the
// root) get `boost * BIAS_WORD_START_FACTOR`; an arc that completes a
// word gets the completion bonus on top of its base; when a token
// continues the match AND starts a new word the larger (continuation)
// value wins. Blank (index width-1) and out-of-range ids are never
// boosted.
class BiasAutomaton {
  public:
    BiasAutomaton() = default;

    // Compile `seqs` (from bias_segment_vocabulary) into an automaton
    // whose rows are `width` floats wide. `width` is the token-class
    // count INCLUDING blank at width-1 (== pred_vocab for TDT/RNNT,
    // n_classes for CTC). boost == 0 or width <= 0 or empty seqs yield
    // an empty() automaton that callers should not consult. Invalid boosts throw
    // invalid_argument; exceeding the dense-row budget throws length_error.
    static BiasAutomaton build(const std::vector<std::vector<int>> & seqs,
                               int                                   width,
                               float                                 boost,
                               const std::unordered_set<int> &       generic_start_ids = {},
                               size_t                                max_dense_bytes   = BIAS_MAX_DENSE_BYTES);

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
