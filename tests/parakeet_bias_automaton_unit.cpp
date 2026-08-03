// parakeet_bias_automaton_unit.cpp - unit tests for the pure vocab-bias
// core (arch/parakeet/bias_automaton.{h,cpp}): greedy longest-match
// segmentation against a cased SentencePiece-style piece table, the
// Aho-Corasick automaton with failure links, and the precomputed dense
// per-state bias rows (word starts + continuations boosted, completing
// arcs get the bonus, blank never boosted).
//
// The cases mirror the reference implementation's spec suite one-for-one
// so the port stays behaviorally aligned; the mini vocabulary below is
// the same table the reference tests use.

#include "arch/parakeet/bias_automaton.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

using transcribe::parakeet::BiasAutomaton;
using transcribe::parakeet::BiasPieceIndex;
using transcribe::parakeet::bias_segment_vocabulary;
using transcribe::parakeet::bias_segment_word;
using transcribe::parakeet::bias_word_variants;
using transcribe::parakeet::build_bias_piece_index;
using transcribe::parakeet::BIAS_COMPLETION_BONUS_FACTOR;

// A miniature cased SentencePiece-style vocabulary. "▁" marks word starts.
const std::vector<std::string> k_vocab = {
    "\xE2\x96\x81",         // 0  ▁
    "\xE2\x96\x81"
    "Char",                 // 1  ▁Char
    "ge",                   // 2
    "Bee",                  // 3
    "\xE2\x96\x81"
    "char",                 // 4  ▁char
    "bee",                  // 5
    "\xE2\x96\x81"
    "link",                 // 6  ▁link
    "er",                   // 7
    "d",                    // 8
    "\xE2\x96\x81"
    "Link",                 // 9  ▁Link
    "a",                    // 10
    "b",                    // 11
    "c",                    // 12
    "\xE2\x96\x81"
    "the",                  // 13 ▁the
};

std::vector<int> seg(const std::string & word) {
    static const BiasPieceIndex idx = build_bias_piece_index(k_vocab);
    return bias_segment_word(word, idx);
}

bool contains_seq(const std::vector<std::vector<int>> & seqs, const std::vector<int> & want) {
    return std::find(seqs.begin(), seqs.end(), want) != seqs.end();
}

bool feq(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

// ---------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------

void test_segment_greedy_longest_match() {
    CHECK(seg("ChargeBee") == (std::vector<int>{ 1, 2, 3 }));  // ▁Char + ge + Bee
}

void test_segment_lowercase_variant() {
    CHECK(seg("chargebee") == (std::vector<int>{ 4, 2, 5 }));  // ▁char + ge + bee
}

void test_segment_falls_back_to_bare_word_start_piece() {
    // No "▁erd"-style piece exists, but the standalone "▁" piece (present
    // in real SentencePiece vocabs too) legitimately spells the word
    // start: ▁ + er + d.
    CHECK(seg("erd") == (std::vector<int>{ 0, 7, 8 }));
}

void test_segment_unspellable_returns_empty() {
    CHECK(seg("Z\xC3\xBCrich").empty());  // "Zürich": no umlaut pieces in the mini vocab
}

void test_word_variants_dedup_and_case() {
    CHECK(bias_word_variants("ChargeBee") == (std::vector<std::string>{ "ChargeBee", "chargebee", "Chargebee" }));
    CHECK(bias_word_variants("linkerd") == (std::vector<std::string>{ "linkerd", "Linkerd" }));
    CHECK(bias_word_variants("  ").empty());
}

void test_segment_vocabulary_collects_variants_and_failures() {
    const auto res = bias_segment_vocabulary({ "Linkerd", "qqq" }, k_vocab);
    // linkerd → ▁link+er+d; Linkerd → ▁Link+er+d (both variants spellable)
    CHECK(contains_seq(res.seqs, { 6, 7, 8 }));
    CHECK(contains_seq(res.seqs, { 9, 7, 8 }));
    CHECK(res.failed == (std::vector<std::string>{ "qqq" }));
}

void test_segment_vocabulary_dedups_identical_sequences() {
    const auto res = bias_segment_vocabulary({ "the", "The", "THE" }, k_vocab);
    // Only "▁the" [13] is spellable; identical variant sequences collapse.
    CHECK(res.seqs == (std::vector<std::vector<int>>{ { 13 } }));
}

// ---------------------------------------------------------------------------
// Automaton
// ---------------------------------------------------------------------------

BiasAutomaton make_automaton(float boost) {
    const auto res = bias_segment_vocabulary({ "ChargeBee", "Linkerd" }, k_vocab);
    CHECK(res.failed.empty());
    return BiasAutomaton::build(res.seqs, static_cast<int>(k_vocab.size()) + 1, boost);
}

void test_advance_follows_goto_arcs_to_terminal() {
    const BiasAutomaton a = make_automaton(2.0f);
    int                 s = a.advance(a.root(), 1);  // ▁Char
    CHECK(a.depth(s) == 1);
    CHECK(!a.terminal(s));
    s = a.advance(s, 2);  // ge
    CHECK(a.depth(s) == 2);
    s = a.advance(s, 3);  // Bee
    CHECK(a.terminal(s));
}

void test_advance_dead_partial_falls_back() {
    const BiasAutomaton a = make_automaton(2.0f);
    int                 s = a.advance(a.root(), 1);  // ▁Char (live partial)
    s                     = a.advance(s, 8);         // "d" kills the match
    CHECK(a.depth(s) <= 1);
    CHECK(!a.terminal(s));
}

void test_advance_fallback_can_start_new_word() {
    const BiasAutomaton a = make_automaton(2.0f);
    int                 s = a.advance(a.root(), 1);  // partial ▁Char…
    s                     = a.advance(s, 6);         // ▁link — a fresh word start
    CHECK(a.depth(s) == 1);  // fell back and re-entered via root's child
}

void test_rows_boost_continuation_wordstarts_and_completion() {
    const float         boost = 2.0f;
    const BiasAutomaton a     = make_automaton(boost);
    const int           width = a.width();

    // At root: word starts only (ids 1, 4, 6, 9), all at `boost`.
    const float * at_root = a.row(a.root());
    for (int t = 0; t < width; ++t) {
        const bool is_start = (t == 1 || t == 4 || t == 6 || t == 9);
        CHECK(feq(at_root[t], is_start ? boost : 0.0f));
    }

    // Two deep (▁Char ge ·): completing arc gets the bonus; word starts
    // still present.
    const int     two_deep = a.advance(a.advance(a.root(), 1), 2);
    const float * row      = a.row(two_deep);
    CHECK(feq(row[3], boost * (1.0f + BIAS_COMPLETION_BONUS_FACTOR)));
    CHECK(feq(row[1], boost));
}

void test_row_dense_and_never_boosts_blank() {
    const BiasAutomaton a     = make_automaton(2.0f);
    const int           width = a.width();
    CHECK(width == static_cast<int>(k_vocab.size()) + 1);
    const float * row = a.row(a.root());
    CHECK(feq(row[width - 1], 0.0f));  // blank
    CHECK(feq(row[1], 2.0f));
    CHECK(feq(row[0], 0.0f));
    // Precomputed: same pointer on a second call.
    CHECK(a.row(a.root()) == row);
}

void test_state_isolated_per_decode_call() {
    // Fresh decode = start at root; state is a caller-owned integer, so
    // no leakage between calls by construction. Smoke: states distinct.
    const BiasAutomaton a = make_automaton(2.0f);
    CHECK(a.advance(a.root(), 1) != a.root());
}

void test_empty_automaton_is_safe() {
    const BiasAutomaton a = BiasAutomaton::build({}, static_cast<int>(k_vocab.size()) + 1, 1.0f);
    CHECK(a.empty());
    CHECK(a.advance(a.root(), 5) == a.root());
    const float * row = a.row(a.root());
    for (int t = 0; t < a.width(); ++t) {
        CHECK(feq(row[t], 0.0f));
    }
}

void test_out_of_range_state_is_treated_as_root() {
    const BiasAutomaton a = make_automaton(2.0f);
    CHECK(a.row(9999) == a.row(a.root()));
    CHECK(a.advance(9999, 1) == a.advance(a.root(), 1));
}

}  // namespace

int main() {
    test_segment_greedy_longest_match();
    test_segment_lowercase_variant();
    test_segment_falls_back_to_bare_word_start_piece();
    test_segment_unspellable_returns_empty();
    test_word_variants_dedup_and_case();
    test_segment_vocabulary_collects_variants_and_failures();
    test_segment_vocabulary_dedups_identical_sequences();
    test_advance_follows_goto_arcs_to_terminal();
    test_advance_dead_partial_falls_back();
    test_advance_fallback_can_start_new_word();
    test_rows_boost_continuation_wordstarts_and_completion();
    test_row_dense_and_never_boosts_blank();
    test_state_isolated_per_decode_call();
    test_empty_automaton_is_safe();
    test_out_of_range_state_is_treated_as_root();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("parakeet_bias_automaton_unit: all checks passed\n");
    return 0;
}
