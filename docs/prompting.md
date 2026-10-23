# Prompting

Generic prompting fields on `transcribe_run_params` (CLI flag in parentheses).
Probe `transcribe_model_supports()` for the matching feature bit.

| Field | Feature bit | Effect |
|---|---|---|
| `vocabulary` (`--vocabulary`, `--vocabulary-file`) | `VOCABULARY` | Custom terms, priority order, formatted per family. |
| `prompt` (`--prompt`) | `CONTEXT_PROMPT` | Context text in the model's conditioning slot. |
| `task = INSTRUCT` + `prompt` (`--task instruct`) | `INSTRUCT` | `prompt` replaces the transcription request; output is free text. |
| `prefix` (`--prefix`) | `TRANSCRIPT_PREFIX` | Transcript text the model continues from. Unsupported is an error. |

| Family | Vocabulary | Context prompt | Instruct | Prefix |
|---|---|---|---|---|
| Parakeet | yes | | | |
| Whisper | yes | yes | | yes |
| Qwen3-ASR | yes | yes | | |
| Voxtral (2507) | | | yes | |
| Granite 4.0-1b / 4.1-2b | yes | | | |
| Granite 4.1-2b-plus | yes | | | yes |
| Canary 180m-flash / 1b-flash / 1b-v2 | | | | yes |
| Fun-ASR-Nano | yes | | | |
| MOSS-Transcribe-Diarize | yes | | | |

Per-model formats and restrictions are in each model doc's **Prompting:**
note under [`models/`](models/).

## Limits

- Over the prompt budget, `vocabulary` drops terms from the end of the list and
  `prompt` keeps its most recent text, both with a WARN. An INSTRUCT prompt
  that does not fit is an error.
- INSTRUCT requires `target_language == NULL` and timestamps NONE or AUTO.
- `prefix`: the audio must contain the prefix's speech, and long-form models
  apply it to the first window only.
- A feature bit covers plain transcription. Under another task or output mode
  a model may ignore `vocabulary` or `prompt` with a WARN; its model doc says
  when.

Parakeet applies vocabulary terms through decoder-level biasing in offline and
streaming recognition. The default boost is 3.0; optional Parakeet extensions
(or CLI `--bias-boost`) override the strength. An empty list or zero boost
leaves decoding unchanged. Terms that cannot be spelled are skipped with a warning.

Parakeet vocabulary limits: at most 256 input terms, at most 256 UTF-8 bytes per
term, and at most 64 MiB of compiled dense bias rows. Invalid UTF-8, exceeded
limits, and boost values outside the finite range [0, 10] return INVALID_ARG.
Duplicates are deduplicated after segmentation; empty terms are ignored.
Input count, term length, UTF-8 and boost are validated even when boost is zero.
The dense-row budget applies only when compiling an active vocabulary.

The spelling compiler matches SentencePiece-style pieces and normalizes ASCII
whitespace between phrase words. It finds complete paths rather than stopping
at greedy dead ends; case variants are ASCII-only. This is not exact source
SentencePiece encoding or full Unicode normalization: older Parakeet GGUFs
label both rank-based and probability-scored tokenizers as `bpe`. Unspellable
variants remain skipped with a warning.
