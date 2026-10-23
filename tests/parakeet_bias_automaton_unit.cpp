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
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

using transcribe::parakeet::BIAS_COMPLETION_BONUS_FACTOR;
using transcribe::parakeet::bias_segment_vocabulary;
using transcribe::parakeet::bias_segment_word;
using transcribe::parakeet::BIAS_WORD_START_FACTOR;
using transcribe::parakeet::bias_word_variants;
using transcribe::parakeet::BiasAutomaton;
using transcribe::parakeet::BiasPieceIndex;
using transcribe::parakeet::build_bias_piece_index;

// A miniature cased SentencePiece-style vocabulary. "▁" marks word starts.
const std::vector<std::string> k_vocab = {
    "\xE2\x96\x81",  // 0  ▁
    "\xE2\x96\x81"
    "Char",          // 1  ▁Char
    "ge",            // 2
    "Bee",           // 3
    "\xE2\x96\x81"
    "char",          // 4  ▁char
    "bee",           // 5
    "\xE2\x96\x81"
    "link",          // 6  ▁link
    "er",            // 7
    "d",             // 8
    "\xE2\x96\x81"
    "Link",          // 9  ▁Link
    "a",             // 10
    "b",             // 11
    "c",             // 12
    "\xE2\x96\x81"
    "the",           // 13 ▁the
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
    CHECK(a.depth(s) == 1);                          // fell back and re-entered via root's child
}

void test_generic_start_ids_flag_bare_marker_pieces() {
    // "erd" spells as ▁ + er + d — a bare-marker first piece is a generic
    // start; "Linkerd" begins with the specific "▁link"/"▁Link" pieces.
    const auto res = bias_segment_vocabulary({ "erd", "Linkerd" }, k_vocab);
    CHECK(res.failed.empty());
    CHECK(res.generic_start_ids == (std::unordered_set<int>{ 0 }));
}

void test_generic_start_arcs_get_no_standing_boost() {
    // A generic word-start arc contributes no standing boost anywhere; the
    // match can still begin through it (advance works), and its onward
    // continuations keep full boost. Mirrors the reference suite.
    const auto res = bias_segment_vocabulary({ "erd", "ChargeBee" }, k_vocab);
    CHECK(res.failed.empty());
    const float         boost = 2.0f;
    const BiasAutomaton a =
        BiasAutomaton::build(res.seqs, static_cast<int>(k_vocab.size()) + 1, boost, res.generic_start_ids);
    const float * at_root = a.row(a.root());
    CHECK(feq(at_root[0], 0.0f));                            // generic "▁" arc unboosted
    CHECK(feq(at_root[1], boost * BIAS_WORD_START_FACTOR));  // "▁Char" stays boosted
    const int s = a.advance(a.root(), 0);                    // enter via the generic arc
    CHECK(a.depth(s) == 1);
    CHECK(feq(a.row(s)[7], boost));                          // "er" continuation, full boost
}

void test_rows_boost_continuation_wordstarts_and_completion() {
    const float         boost = 2.0f;
    const BiasAutomaton a     = make_automaton(boost);
    const int           width = a.width();

    // At root: word starts only (ids 1, 4, 6, 9), at the down-weighted
    // word-start base.
    const float * at_root = a.row(a.root());
    for (int t = 0; t < width; ++t) {
        const bool is_start = (t == 1 || t == 4 || t == 6 || t == 9);
        CHECK(feq(at_root[t], is_start ? boost * BIAS_WORD_START_FACTOR : 0.0f));
    }

    // Two deep (▁Char ge ·): completing CONTINUATION arc gets the bonus
    // on the full base; fresh word starts stay at the down-weighted base.
    const int     two_deep = a.advance(a.advance(a.root(), 1), 2);
    const float * row      = a.row(two_deep);
    CHECK(feq(row[3], boost * (1.0f + BIAS_COMPLETION_BONUS_FACTOR)));
    CHECK(feq(row[1], boost * BIAS_WORD_START_FACTOR));
}

void test_word_start_arcs_weigh_less_than_continuations() {
    // The standing word-start mass scales with vocabulary size, so fresh
    // starts are down-weighted; a live match's continuations keep full
    // boost, and a token that is both takes the larger (continuation)
    // value. Mirrors the reference suite's weighting test.
    const float         boost    = 2.0f;
    const BiasAutomaton a        = make_automaton(boost);
    const int           one_deep = a.advance(a.root(), 1);  // ▁Char ·
    const float *       row      = a.row(one_deep);
    CHECK(feq(row[2], boost));                              // "ge" continues the match
    CHECK(feq(row[4], boost * BIAS_WORD_START_FACTOR));     // "▁char" would start anew
    CHECK(row[2] > row[4]);
}

void test_row_dense_and_never_boosts_blank() {
    const BiasAutomaton a     = make_automaton(2.0f);
    const int           width = a.width();
    CHECK(width == static_cast<int>(k_vocab.size()) + 1);
    const float * row = a.row(a.root());
    CHECK(feq(row[width - 1], 0.0f));  // blank
    CHECK(feq(row[1], 2.0f * BIAS_WORD_START_FACTOR));
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

void test_unicode_single_character_start_is_generic() {
    const auto segmented = bias_segment_vocabulary({ "\xD0\xB8x" }, { "\xE2\x96\x81\xD0\xB8", "x" });
    CHECK(segmented.failed.empty());
    CHECK(segmented.generic_start_ids == (std::unordered_set<int>{ 0 }));
}

void test_phrase_boundaries_and_utf8() {
    const std::vector<std::string> pieces = { "\xE2\x96\x81New", "\xE2\x96\x81York", "\xE2\x96\x81Z\xC3\xBCrich" };
    const auto                     index  = build_bias_piece_index(pieces);
    CHECK(bias_segment_word("New York", index) == (std::vector<int>{ 0, 1 }));
    CHECK(bias_segment_word(" \tNew  \nYork \r", index) == (std::vector<int>{ 0, 1 }));
    CHECK(bias_segment_word("Z\xC3\xBCrich", index) == (std::vector<int>{ 2 }));
}

void test_longest_piece_must_leave_spellable_suffix() {
    const auto index =
        build_bias_piece_index({ "\xE2\x96\x81"
                                 "ab",
                                 "\xE2\x96\x81"
                                 "a",
                                 "bc" });
    CHECK(bias_segment_word("abc", index) == (std::vector<int>{ 1, 2 }));
}

void test_overlapping_suffix_continuations() {
    const auto automaton = BiasAutomaton::build(
        {
            { 1, 2, 3 },
            { 2, 4 }
    },
        6, 2.0f, { 2 });
    const int state = automaton.advance(automaton.advance(0, 1), 2);
    CHECK(feq(automaton.row(state)[4], 2.0f * (1.0f + BIAS_COMPLETION_BONUS_FACTOR)));
    CHECK(automaton.terminal(automaton.advance(state, 4)));
    // A generic root token is still allowed full continuation boost when it
    // continues a suffix match, rather than starting a new phrase.
    const auto generic = BiasAutomaton::build(
        {
            { 1, 2, 3 },
            { 2, 2, 4 }
    },
        6, 2.0f, { 2 });
    const int overlapping = generic.advance(generic.advance(0, 1), 2);
    CHECK(feq(generic.row(overlapping)[2], 2.0f));
    CHECK(feq(generic.row(0)[2], 0.0f));
}

void test_memory_budget_and_boost_validation() {
    // Three states require exactly 3 * 4 floats. Reject before allocating.
    CHECK(!BiasAutomaton::build(
               {
                   { 1, 2 }
    },
               4, 3.0f, {}, 48)
               .empty());
    bool rejected = false;
    try {
        (void) BiasAutomaton::build(
            {
                { 1, 2 }
        },
            4, 3.0f, {}, 47);
    } catch (const std::length_error &) {
        rejected = true;
    }
    CHECK(rejected);
    BiasAutomaton::validate_size(
        {
            { 1, 2 },
            { 1, 2 }
    },
        4, 48);
    rejected = false;
    try {
        BiasAutomaton::validate_size(
            {
                { 1, 2 }
        },
            4, 47);
    } catch (const std::length_error &) {
        rejected = true;
    }
    CHECK(rejected);
    CHECK(transcribe::parakeet::bias_normalize_term(" \tNew\f\v York\r\n") == "New York");
    for (float boost :
         { -1.0f, 10.1f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() }) {
        rejected = false;
        try {
            (void) BiasAutomaton::build(
                {
                    { 1, 2 }
            },
                4, boost);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        CHECK(rejected);
    }
    CHECK(BiasAutomaton::build(
              {
                  { 1, 2 }
    },
              4, 0.0f)
              .empty());
}

}  // namespace

int main() {
    test_unicode_single_character_start_is_generic();
    test_phrase_boundaries_and_utf8();
    test_longest_piece_must_leave_spellable_suffix();
    test_overlapping_suffix_continuations();
    test_memory_budget_and_boost_validation();
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
    test_generic_start_ids_flag_bare_marker_pieces();
    test_generic_start_arcs_get_no_standing_boost();
    test_rows_boost_continuation_wordstarts_and_completion();
    test_word_start_arcs_weigh_less_than_continuations();
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
