# Nemotron-Labs Unified ASR 0.6B

Upstream: [`nvidia/nemotron-labs-unified-asr-0.6b`](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b)
at `023ce907267a09a308a5cb69239447cb3d0c1597`. Licensed [OpenMDW-1.1](https://openmdw.ai/license/1-1/).

> **Status: fork-local port, not published.** There is no canonical GGUF, catalog record or WER table
> for this variant yet. Everything under [Validation](#validation) was measured on a locally converted
> F32 GGUF; anything not listed there has not been run.

## What it is

Multilingual speech-to-text (25 European languages) with punctuation and capitalization. It is
[`parakeet-unified-en-0.6b`](parakeet-unified-en-0.6b.md) with two changes:

- a 4095-piece SentencePiece vocabulary shared by all languages (Latin, Cyrillic and Greek pieces),
  instead of the 1024-piece English one;
- the language prompt layer of
  [`nemotron-3.5-asr-streaming-0.6b`](nemotron-3.5-asr-streaming-0.6b.md): a 128-slot one-hot is
  concatenated to every encoder output frame and passed through a two-layer MLP before the joint.

The encoder, predictor and joint are otherwise tensor-for-tensor the Unified EN architecture, so the
port runs it in the same two modes from one GGUF:

- **Offline** (`transcribe_run`): one full-attention pass over the audio.
- **Buffered streaming** (`transcribe_stream_{begin,feed,finalize}`): each step re-runs the encoder over
  a sliding `[left | chunk | right]` window with the `chunked_limited_with_rc` mask and carries the
  RNN-T decoder state across steps. This is not cache-aware streaming; cost per second of audio grows
  as the chunk shrinks.

## Languages

`general.languages` lists the 25 short codes:
`bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro ru sk sl es sv uk`.

- Pass one of them as `language` (`-l de`). Each is an exact key of the model's prompt dictionary.
- `language = NULL` selects the dictionary's `auto` slot (127). The model transcribes, but it has no
  language-tag tokens, so it **cannot report which language it heard**:
  `stt.capability.lang_detect` is false and `transcribe_detected_language` stays empty.
- The prompt dictionary in the GGUF also carries locale keys (`en-US`, `es-ES`, `pt-BR`, …). Some
  select a *different* trained slot than the short code (`es` → 3 but `es-ES` → 2). They are not
  reachable through the public API, which rejects any hint that is not in `general.languages`.

## Streaming geometry

Same trained menu as Unified EN, in 80 ms encoder frames: `L = 70`, `C ∈ {1, 2, 7, 13}`,
`R ∈ {0, 1, 2, 3, 4, 7, 13}`. Default `(70, 13, 13)` = 5.6 s / 1.04 s / 1.04 s (2.08 s lookahead).

```bash
# 1.12 s lookahead
transcribe-cli -m nemotron-labs-unified-asr-0.6b-Q8_0.gguf -l en \
    --stream-chunk-ms 500 --stream-buf-chunk-ms 560 --stream-buf-right-ms 560 audio.wav
```

The model card's 0.56 s row is `(70, 2, 5)`. `R = 5` is outside the trained menu and is rejected;
use `(70, 2, 4)` (480 ms) or `(70, 7, 7)` (1.12 s).

## Converting

```bash
# the output directory name selects the converter profile
uv run --project scripts/envs/parakeet scripts/convert-parakeet.py /path/to/nemotron-labs-unified-asr-0.6b.nemo \
    models/nemotron-labs-unified-asr-0.6b/nemotron-labs-unified-asr-0.6b-F32.gguf \
    --repo-id nvidia/nemotron-labs-unified-asr-0.6b
transcribe-quantize …-F32.gguf …-Q8_0.gguf --quant Q8_0
```

Source `.nemo`: 2,507,735,040 bytes, sha256
`1718cc1b44fb8597b2356328d706a8d89f31c389ca6f2256c2e3f743c0246936`. The conversion writes 965 tensors
(Unified EN's 961 plus the four prompt tensors).

## Validation

Measured 2026-11-09 on a locally converted F32 GGUF (sha256 `968df137…b5648b`), CPU backend, 1 thread,
against NeMo 3.0.0 (`EncDecRNNTBPEModelWithPrompt`). No WER has been measured.

**Conversion.** Seven tensors (all prompt tensors, predictor embedding, joint output) compared with the
`.nemo` state dict: max |diff| 0. Preflight gate B passes.

**Offline** (`scripts/dump_reference_parakeet_nemo.py encoder`/`decode --offline-only` vs
`transcribe-cli`, `scripts/compare_tensors.py` with `tests/tolerances/nemotron-labs-unified-asr-0.6b.json`):

| Clip | Language (slot) | Transcript vs NeMo | `enc.final` max abs | `enc.prompted` max abs |
| --- | --- | --- | ---: | ---: |
| jfk | en (0) | identical | 4.5e-3 | 1.8e-2 |
| jfk | NULL / auto (127) | identical | 4.5e-3 | 1.5e-2 |
| fleurs-de | de (9) | identical | 3.2e-3 | 1.3e-2 |
| fleurs-es | es (3) | identical | 7.3e-4 | 5.1e-3 |

**Buffered streaming** (`scripts/validate_buffered_streaming.py --language …`). Every row: same chunk
steps on both sides, PCM windows bit-identical, the uploaded one-hot hot at the expected slot on every
frame, and the prompt layer reproduced in NumPy from the C++ pre-prompt rows.

| Clip | Language (slot) | (L, C, R) | Chunks | Prompt replay max abs | `enc_out` worst mean abs | Transcript vs NeMo |
| --- | --- | --- | ---: | ---: | ---: | --- |
| jfk | en (0) | 70, 13, 13 | 10 | 9.7e-7 | 5.7e-3 | identical |
| jfk | auto (127) | 70, 13, 13 | 10 | 7.8e-7 | 5.7e-3 | identical |
| fleurs-de | de (9) | 70, 13, 13 | 10 | 9.2e-7 | 4.7e-3 | identical |
| fleurs-es | es (3) | 70, 13, 13 | 9 | 9.1e-7 | 1.4e-2 | + final `.` |
| german (29 s) | de (9) | 70, 13, 13 | 28 | 8.8e-7 | 8.5e-3 | identical |
| fleurs-de | auto (127) | 70, 7, 7 | 19 | 9.0e-7 | 4.3e-3 | identical |
| jfk | en (0) | 70, 7, 7 | 19 | 7.9e-7 | 4.2e-3 | identical |
| dots (35 s) | en (0) | 70, 7, 7 | 63 | 7.4e-7 | 6.9e-3 | identical |
| jfk | en (0) | 70, 2, 4 | 67 | 9.3e-7 | 8.7e-3 | + final `.` |
| fleurs-es | es (3) | 70, 2, 4 | 59 | 7.5e-7 | 1.6e-2 | identical |
| jfk | en (0) | 70, 1, 1 | 137 | 7.4e-7 | 2.0e-2 | + final `.` |

"+ final `.`" is the C++ end-of-stream silence lookahead, which NeMo's loop does not have; it also
raises `enc_out` drift on the last chunk.

Negative controls on jfk / en / (70, 13, 13), both of which must and do fail:

| Control | One-hot check | Prompt replay max abs | Transcript |
| --- | --- | ---: | --- |
| Build without the one-hot upload | missing | not computable | unchanged |
| C++ run with `de` against an `en` reference | slot 9 on all frames | 2.2e-1 | unchanged |

The unchanged transcripts on this clip are why the prompt path has its own tensor-level gates.

**Other checks.** `transcribe_parakeet_stream_ext_reject_unit` (language pre-flight on toy prompt
fixtures) and `transcribe_parakeet_buffered_stream_eos_smoke` (flush at the EOS cut and at an exact
`C+R+k*C` boundary, `language = NULL`) pass on this model. Q8_0 (sha256 `cc5b9a34…da38542d`,
740 MB) transcribes jfk identically to F32, offline and streaming. Vocabulary biasing compiles and
runs with 200 terms.

**Not validated.**

- WER on any corpus, any quantization.
- `(70, 1, 0)` (zero lookahead): the harness reports a row-count mismatch against the reference at
  `R = 0`. The same mismatch occurs on `parakeet-unified-en-0.6b` with an unmodified build, so it is
  not specific to this variant, but this geometry is unverified here. The prompt replay still holds
  on all 138 chunks.
- `scripts/validate.py` through the pinned `scripts/envs/parakeet` environment (NeMo `6967f48`); the
  comparisons above ran the same dumper and comparator by hand under NeMo 3.0.0.
- NeMo's own `speech_to_text_streaming_infer_rnnt.py --target_lang` (NeMo ≥ `b19b1a79`); the reference
  here is the dumper's loop, which mirrors that script and now passes `prompt_indices`.
- Metal and other GPU backends, batch decoding with mixed languages, real-time performance.

## Known limits

- No detected language in auto mode (see [Languages](#languages)).
- Offline has no long-form chunking (the model has no VAD head); memory grows with the square of the
  audio length.
- English is split into more pieces than with Unified EN's vocabulary, so decoding takes more steps
  per word and custom vocabulary terms compile into longer paths.
- On the tested English clip (jfk) a missing or wrong prompt slot left the transcript unchanged, so
  transcript checks cannot be relied on to validate the prompt path. Use `scripts/validate_buffered_streaming.py --language`,
  which inspects the uploaded one-hot and compares the encoder output before and after the prompt
  layer.
