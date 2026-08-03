/*
 * include/transcribe/parakeet.h - Parakeet-family public extension surface.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. Holds the
 * streaming extension structs (cache-aware and chunked-attention
 * variants), the run-slot vocab-bias extension, and their kind
 * constants and init functions.
 *
 * Acceptance is per-loaded-model-variant on the STREAM slot:
 * nemotron-speech-streaming-en-0.6b (cache-aware) accepts
 * TRANSCRIBE_EXT_KIND_PARAKEET_STREAM and rejects
 * TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM; parakeet-unified-en-0.6b
 * (chunked_limited_with_rc) does the opposite. On the RUN slot every
 * parakeet variant (TDT / RNN-T / CTC heads alike) accepts
 * TRANSCRIBE_EXT_KIND_PARAKEET_BIAS_RUN. Probe via
 * transcribe_model_accepts_ext_kind before pointing
 * transcribe_stream_params::family / transcribe_run_params::family at
 * one of these structs.
 *
 * FourCC kinds are reserved in docs/extension-kinds.md.
 */

#ifndef TRANSCRIBE_PARAKEET_H
#define TRANSCRIBE_PARAKEET_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'PKST' little-endian = 0x54534B50 */
#define TRANSCRIBE_EXT_KIND_PARAKEET_STREAM          0x54534B50u
/* 'PKBS' little-endian = 0x53424B50 */
#define TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM 0x53424B50u
/* 'PKBR' little-endian = 0x52424B50 */
#define TRANSCRIBE_EXT_KIND_PARAKEET_BIAS_RUN        0x52424B50u

/*
 * Custom-vocabulary biasing (word boosting / shallow fusion), shared
 * field block. Parakeet has no word vocabulary to extend — words are
 * spelled from a fixed set of SentencePiece subword pieces — so
 * "custom vocabulary" means boosting, at greedy-decode time, the token
 * sequences that spell the caller's words (NVIDIA "TurboBias"-style
 * phrase boosting reduced to greedy decoding: an Aho-Corasick automaton
 * over the emitted tokens adds `boost` logit units to continuations of
 * a live partial match and to word-start tokens, plus a completion
 * bonus on word-final arcs; blank, the TDT duration head, and reported
 * confidences are never touched).
 *
 *   bias_words / words
 *
 *     Array of `n` UTF-8 word/phrase strings, or NULL for no biasing.
 *     Words are matched in their given casing plus ASCII-folded
 *     lowercase / Capitalized variants; a word that cannot be spelled
 *     with the model's pieces is skipped (with a WARN log). The array
 *     and the strings are copied at the entry point (transcribe_run /
 *     transcribe_stream_begin) — the caller may free them as soon as
 *     the call returns.
 *
 *   bias_boost / boost
 *
 *     Additive logit boost per matched token, in logit units. The init
 *     functions default it to 3.0f, the empirically tuned sweet spot
 *     for parakeet-tdt-0.6b (≈1.8 flips nothing; ≈10 visibly
 *     over-boosts). 0 disables biasing; negative values are accepted
 *     (anti-boost) but untested.
 */

/*
 * Cache-aware streaming knob (nemotron-speech-streaming-en-0.6b).
 *
 *   att_context_right
 *
 *     Right-context (lookahead) selector in encoder frames. The cache-
 *     aware streaming variants are trained on a menu of (left, right)
 *     pairs simultaneously - the user picks one at inference time to
 *     trade latency for accuracy. nemotron's published menu is
 *     right ∈ {13, 6, 1, 0}, corresponding to lookahead of
 *     {1040, 480, 80, 0} ms at the 80ms encoder frame rate.
 *
 *     -1 (default): use the model's default setting (first entry of
 *                   att_context_size_choices = max-accuracy /
 *                   max-latency).
 *     < -1:         caller bug; transcribe_stream_begin returns
 *                   TRANSCRIBE_ERR_INVALID_ARG.
 *     >= 0:         select the corresponding (left, att_context_right)
 *                   entry from the model's training menu. 0 is
 *                   legitimate when 0-frame lookahead is in the menu.
 *                   transcribe_stream_begin returns
 *                   TRANSCRIBE_ERR_INVALID_ARG if the requested right
 *                   is not in the menu.
 *
 *     The published menu and the lookahead (in milliseconds) each
 *     entry corresponds to are documented in the model's family doc
 *     (docs/models/nemotron-speech-streaming-en-0.6b.md); -1 selects
 *     the model's default (max-accuracy / max-latency) entry.
 */
struct transcribe_parakeet_stream_ext {
    struct transcribe_ext ext;
    int32_t               att_context_right;

    /*
     * v2 additive fields — custom-vocabulary biasing (see the shared
     * field-block comment next to TRANSCRIBE_EXT_KIND_PARAKEET_BIAS_RUN).
     * A caller compiled against the v1 layout passes the v1 sizeof in
     * ext.size and the runtime treats these as absent (no biasing).
     * The streaming automaton state persists across chunks within a
     * stream (reset at transcribe_stream_begin), so a custom word split
     * across a chunk boundary keeps its partial match.
     */
    const char * const * bias_words;
    uint64_t             n_bias_words;
    float                bias_boost;
};

/* Fills ext.size/kind, att_context_right = -1 (model default), and the
 * bias fields to NULL/0/3.0f (no biasing, default boost). */
TRANSCRIBE_API void transcribe_parakeet_stream_ext_init(struct transcribe_parakeet_stream_ext * ext);

/*
 * Chunked-attention (buffered) streaming knob (parakeet-unified-en-0.6b).
 *
 * parakeet-unified-en-0.6b is trained with chunked_limited_with_rc
 * attention over a menu of (left, chunk, right) context tuples
 * expressed in 80ms encoder frames. The user picks the active tuple at
 * stream_begin time; the encoder re-runs over each new
 * [left | chunk | right] PCM window. Each field is in MILLISECONDS;
 * the runtime converts to encoder frames at the model's frame rate.
 *
 * Per-field sentinels (each field independently):
 *
 *   -1     "use the model default for this field." Unified-en-0.6b's
 *          best-accuracy default is L=5600 ms / C=1040 ms / R=1040 ms,
 *          which the published WER numbers correspond to.
 *   < -1   caller bug; transcribe_stream_begin returns
 *          TRANSCRIBE_ERR_INVALID_ARG.
 *   0      a legitimate requested value when 0 frames is in the model's
 *          menu for that field. Not all fields admit 0.
 *   > 0    must be an exact positive multiple of the encoder frame size
 *          (80 ms for every shipped FastConformer streaming variant).
 *          A value that does not divide the frame returns
 *          TRANSCRIBE_ERR_INVALID_ARG; the runtime never silently floors.
 *
 * After per-field resolution the (L, C, R) frame tuple is validated
 * against the model's training menu
 * (stt.parakeet.encoder.att_chunk_{left,chunk,right}_choices); tuples
 * outside the menu return TRANSCRIBE_ERR_INVALID_ARG.
 */
struct transcribe_parakeet_buffered_stream_ext {
    struct transcribe_ext ext;
    int32_t               left_ms;
    int32_t               chunk_ms;
    int32_t               right_ms;

    /*
     * v2 additive fields — custom-vocabulary biasing; identical
     * semantics to transcribe_parakeet_stream_ext's bias fields (v1
     * callers pass the v1 sizeof and get no biasing; automaton state
     * persists across chunks, reset at transcribe_stream_begin).
     */
    const char * const * bias_words;
    uint64_t             n_bias_words;
    float                bias_boost;
};

/* Fills ext.size/kind, left/chunk/right_ms = -1 (model default), and
 * the bias fields to NULL/0/3.0f (no biasing, default boost). */
TRANSCRIBE_API void transcribe_parakeet_buffered_stream_ext_init(struct transcribe_parakeet_buffered_stream_ext * ext);

/*
 * Run-slot custom-vocabulary biasing for the offline decode paths
 * (transcribe_run / transcribe_run_batch), accepted by every parakeet
 * variant regardless of head kind (TDT / RNN-T / CTC). Field semantics
 * are the shared bias block documented next to
 * TRANSCRIBE_EXT_KIND_PARAKEET_BIAS_RUN; the automaton state is
 * per-utterance (each run / batch item starts at the automaton root).
 */
struct transcribe_parakeet_bias_run_ext {
    struct transcribe_ext ext;
    const char * const *  words;
    uint64_t              n_words;
    float                 boost;
};

/* Fills ext.size/kind, words = NULL, n_words = 0, boost = 3.0f. */
TRANSCRIBE_API void transcribe_parakeet_bias_run_ext_init(struct transcribe_parakeet_bias_run_ext * ext);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TRANSCRIBE_PARAKEET_H */
