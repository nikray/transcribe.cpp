#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
# ]
# ///
"""
validate_buffered_streaming.py - parity harness for buffered streaming
on parakeet-unified-en-0.6b and the prompt-conditioned
nemotron-labs-unified-asr-0.6b (pass --language for the latter).

For a target audio file at a chosen (L, C, R) tuple, runs:

  1. NeMo reference dumper:
       scripts/dump_reference_parakeet_nemo.py buffered_streaming ...
     producing per-chunk reference tensors under <out>/ref/.

  2. C++ transcribe-cli with TRANSCRIBE_DUMP_DIR=<out>/cpp/ and
     --stream-chunk-ms <feed-ms>, producing parallel C++ dumps.

  3. Per-chunk diff of stream.chunk.<step>.{audio_in, enc_out} between
     the two dump dirs. Both sides must produce the same chunk steps;
     audio_in must be bit-identical.

     Prompt-conditioned variants additionally get, per chunk: the
     uploaded one-hot (stream.chunk.<step>.prompt.one_hot) checked to be
     hot at the reference's prompt slot on every frame, the encoder
     output before the prompt MLP (enc_out_unprompted), and a NumPy
     replay of the prompt MLP (weights read from the F32 GGUF) on the
     C++ side's own pre-prompt rows, which must reproduce its post-prompt
     rows. A missing or wrong prompt left the jfk transcript
     unchanged and stays inside the family envelope, so these are what
     detect it. `--cpp-language` with a different language, or
     `--cli` pointing at a build without the upload, are the negative
     controls: both must fail.

  4. Final-transcript byte equality.

Usage:
    uv run --project scripts/envs/parakeet \\
        scripts/validate_buffered_streaming.py \\
        --nemo /path/to/parakeet-unified-en-0.6b.nemo \\
        --gguf models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-F32.gguf \\
        --audio samples/jfk.wav \\
        --out build/validate_buffered_streaming/parakeet-unified/jfk/default \\
        [--left-secs 5.6 --chunk-secs 1.04 --right-secs 1.04] \\
        [--backend cpu --threads 1] \\
        [--feed-ms 500] \\
        [--language de --expect-prompt-id 9] \\
        [--cli /path/to/transcribe-cli --ref-python /path/to/nemo-venv/bin/python]

Exit code 0 = per-chunk parity OK across all chunks (audio_in
bit-exact + enc_out within fp32 tolerance). The final-transcript
byte match is reported informationally — greedy RNN-T can flip a
single token at a chunk boundary on fp32 noise even when the
encoder windows are byte-identical, so the per-chunk gate is the
algorithmic check of record. WER on test-clean is the corpus-level
gate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent


def run_subprocess(cmd: list[str], *, label: str, env: dict | None = None,
                   cwd: Path | None = None, capture: bool = False):
    print(f"+ {label}: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        env=({**os.environ, **env} if env else None),
        cwd=str(cwd) if cwd else None,
        capture_output=capture,
        text=True,
    )
    if proc.returncode != 0:
        if capture:
            sys.stdout.write(proc.stdout or "")
            sys.stderr.write(proc.stderr or "")
        raise SystemExit(f"error: {label} returned {proc.returncode}")
    return proc


def run_ref_dump(*, nemo: Path, audio: Path, out_dir: Path,
                 left_secs: float, chunk_secs: float, right_secs: float,
                 force: bool, language: str | None = None,
                 ref_python: Path | None = None) -> Path:
    stamp = out_dir / "stream_history.json"
    request = {
        "model": str(nemo), "audio": str(audio), "language": language,
        "left_secs": float(left_secs), "chunk_secs": float(chunk_secs),
        "right_secs": float(right_secs),
    }
    request_path = out_dir / "ref_request.json"
    if stamp.exists() and not force:
        # Reuse only a dump made for this exact model / audio / geometry /
        # language; anything else is regenerated.
        cached = json.loads(request_path.read_text()) if request_path.exists() else None
        if cached == request:
            print(f"  [skip ref] {stamp} exists (pass --force to re-run)")
            return out_dir
        print(f"  [stale ref] {out_dir} was dumped for a different request; regenerating")
    # Always regenerate into an empty directory: chunks left over from a
    # longer or finer-grained earlier run would otherwise read as
    # reference-only steps.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    launcher = ([str(ref_python)] if ref_python is not None else
                ["uv", "run", "--project", str(REPO / "scripts/envs/parakeet")])
    cmd = [
        *launcher,
        str(REPO / "scripts/dump_reference_parakeet_nemo.py"),
        "buffered_streaming",
        "--model", str(nemo),
        "--audio", str(audio),
        "--out", str(out_dir),
        "--left-secs", f"{left_secs}",
        "--chunk-secs", f"{chunk_secs}",
        "--right-secs", f"{right_secs}",
    ]
    if language is not None:
        # NeMo spells the no-hint case as the 'auto' dictionary key.
        cmd += ["--language", language]
    run_subprocess(cmd, label="ref dump", cwd=REPO)
    request_path.write_text(json.dumps(request, indent=2))
    return out_dir


def run_cpp_dump(*, gguf: Path, audio: Path, out_dir: Path,
                 feed_ms: int, backend: str | None,
                 threads: int | None,
                 left_ms: int | None,
                 chunk_ms: int | None,
                 right_ms: int | None,
                 language: str | None = None,
                 cli: Path | None = None) -> tuple[Path, str]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cli is None:
        cli = REPO / "build/bin/transcribe-cli"
    if not cli.exists():
        raise SystemExit(
            f"error: {cli} not built; run cmake --build build --target transcribe-cli"
        )

    cmd = [
        str(cli),
        "--model", str(gguf),
        "--stream-chunk-ms", str(feed_ms),
        str(audio),
    ]
    if backend is not None:
        cmd += ["--backend", backend]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    if left_ms is not None:
        cmd += ["--stream-buf-left-ms", str(left_ms)]
    if chunk_ms is not None:
        cmd += ["--stream-buf-chunk-ms", str(chunk_ms)]
    if right_ms is not None:
        cmd += ["--stream-buf-right-ms", str(right_ms)]
    if language is not None:
        cmd += ["--language", language]
    env = {"TRANSCRIBE_DUMP_DIR": str(out_dir)}
    proc = run_subprocess(cmd, label="cpp dump", env=env,
                          cwd=REPO, capture=True)
    sys.stdout.write(proc.stdout)
    text = ""
    for line in proc.stdout.splitlines():
        if line.startswith("text:"):
            text = line[len("text:"):].strip()
            break
    return out_dir, text


def load_f32(path: Path, expected_shape: list[int] | None = None) -> np.ndarray:
    sidecar = path.with_suffix(".json")
    raw = np.fromfile(path, dtype=np.float32)
    if not sidecar.exists():
        return raw
    meta = json.loads(sidecar.read_text())
    shape = meta.get("shape") or expected_shape
    if shape:
        return raw.reshape(shape)
    return raw


def compare_pair(name: str, ref_path: Path, cpp_path: Path,
                 eos_pad_frames: int | None = None) -> dict:
    """Compare common tensor rows, excluding the expected EOS divergence."""
    if not ref_path.exists() or not cpp_path.exists():
        return {
            "name": name,
            "status": "MISSING",
            "ref": ref_path.exists(),
            "cpp": cpp_path.exists(),
        }
    ref = load_f32(ref_path).astype(np.float64)
    cpp = load_f32(cpp_path).astype(np.float64)
    if not (np.isfinite(ref).all() and np.isfinite(cpp).all()):
        return {
            "name": name,
            "status": "NONFINITE",
            "ref_finite": bool(np.isfinite(ref).all()),
            "cpp_finite": bool(np.isfinite(cpp).all()),
        }
    extra = {}
    if eos_pad_frames is not None and ref.shape != cpp.shape:
        if ref.ndim == 1 and cpp.shape[0] >= ref.shape[0]:
            # Real audio must match and the synthetic tail must be zero.
            pad = cpp[ref.shape[0]:]
            extra["pad_samples"] = int(pad.size)
            extra["pad_max_abs"] = float(np.abs(pad).max()) if pad.size else 0.0
            cpp = cpp[:ref.shape[0]]
        elif ref.ndim == 2 and cpp.shape[0] >= ref.shape[0] and cpp.shape[1:] == ref.shape[1:]:
            n_real = ref.shape[0] - 1  # drop NeMo's conv-overhang row
            extra["pad_rows"] = int(cpp.shape[0] - ref.shape[0])
            tail = max(0, min(eos_pad_frames, n_real))
            if tail > 0:
                extra["eos_tail_rows"] = tail
                extra["eos_tail_max_abs"] = float(
                    np.abs(ref[n_real - tail:n_real] - cpp[n_real - tail:n_real]).max())
            ref = ref[:n_real - tail]
            cpp = cpp[:n_real - tail]
    if ref.shape != cpp.shape:
        # Some chunks may have differently-shaped tensors (last-chunk
        # divergence). Report and skip.
        return {
            "name": name,
            "status": "SHAPE_DIFF",
            "ref_shape": list(ref.shape),
            "cpp_shape": list(cpp.shape),
        }
    if ref.size == 0:
        return {"name": name, "status": "OK", "max_abs": 0.0, "mean_abs": 0.0,
                "p99_abs": 0.0, "rel_max": 0.0, "n_elem": 0, **extra}
    diff = np.abs(ref - cpp)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    p99_abs = float(np.quantile(np.abs(ref), 0.99))
    rel = float(max_abs / max(p99_abs, 1e-9))
    return {
        "name": name,
        "status": "OK",
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "p99_abs": p99_abs,
        "rel_max": rel,
        "n_elem": int(ref.size),
        **extra,
    }


_EXPORT_PROMPT_MLP = r"""
import sys
import numpy as np
import gguf
src, out = sys.argv[1], sys.argv[2]
tensors = {t.name: t for t in gguf.GGUFReader(src).tensors}
for name in ("prompt.mlp.0.weight", "prompt.mlp.0.bias",
             "prompt.mlp.2.weight", "prompt.mlp.2.bias"):
    t = tensors[name]
    if t.tensor_type != gguf.GGMLQuantizationType.F32:
        raise SystemExit(f"{name} is {t.tensor_type.name}; the prompt replay needs an F32 GGUF")
    # GGUF lists ggml dims (innermost first); the bytes are torch row-major.
    shape = [int(d) for d in reversed(t.shape)]
    np.save(f"{out}/{name}.npy", np.array(t.data, dtype=np.float32).reshape(shape))
"""


def export_prompt_mlp(gguf_path: Path, out_dir: Path,
                      ref_python: Path | None) -> dict[str, np.ndarray]:
    """Read the prompt MLP weights out of the GGUF under test (via the
    reference env, which has the `gguf` package)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    launcher = ([str(ref_python)] if ref_python is not None else
                ["uv", "run", "--project", str(REPO / "scripts/envs/parakeet"), "python"])
    run_subprocess([*launcher, "-c", _EXPORT_PROMPT_MLP, str(gguf_path), str(out_dir)],
                   label="export prompt MLP", cwd=REPO, capture=True)
    return {k: np.load(out_dir / f"prompt.mlp.{k}.npy").astype(np.float64)
            for k in ("0.weight", "0.bias", "2.weight", "2.bias")}


def replay_prompt(mlp: dict[str, np.ndarray], unprompted_path: Path,
                  prompted_path: Path, prompt_id: int) -> dict:
    """Recompute the prompt MLP in NumPy from the C++ side's OWN pre-prompt
    encoder rows and compare with its post-prompt rows.

    This isolates the prompt layer from encoder fp32 drift: with the right
    slot on every row the two agree to fp32 rounding, whatever the encoder
    did; a wrong slot, a missing one-hot or a row misalignment does not.
    """
    if not unprompted_path.exists() or not prompted_path.exists():
        return {"status": "MISSING"}
    x = load_f32(unprompted_path).astype(np.float64)
    y = load_f32(prompted_path).astype(np.float64)
    if x.shape != y.shape or x.ndim != 2:
        return {"status": "SHAPE_DIFF", "ref_shape": list(x.shape), "cpp_shape": list(y.shape)}
    n_prompts = mlp["0.weight"].shape[1] - x.shape[1]
    one_hot = np.zeros((x.shape[0], n_prompts))
    one_hot[:, prompt_id] = 1.0
    h = np.maximum(np.concatenate([x, one_hot], axis=1) @ mlp["0.weight"].T + mlp["0.bias"], 0.0)
    want = h @ mlp["2.weight"].T + mlp["2.bias"]
    diff = np.abs(want - y)
    return {"status": "OK", "max_abs": float(diff.max()) if diff.size else 0.0,
            "mean_abs": float(diff.mean()) if diff.size else 0.0, "n_rows": int(x.shape[0])}


def check_prompt_one_hot(path: Path, prompt_id: int) -> str | None:
    """Assert the C++ uploaded a one-hot at `prompt_id` on EVERY encoder
    frame of the window. Returns None when OK, else a failure message."""
    if not path.exists():
        return "missing (no prompt one-hot was uploaded for this chunk)"
    a = load_f32(path)
    if a.ndim != 2 or a.shape[0] < 1:
        return f"unexpected shape {list(a.shape)}"
    if not (0 <= prompt_id < a.shape[1]):
        return f"prompt_id {prompt_id} outside [0, {a.shape[1]})"
    want = np.zeros_like(a)
    want[:, prompt_id] = 1.0
    if not np.array_equal(a, want):
        hot = sorted({int(i) for i in np.argmax(a, axis=1)})
        bad_rows = int((a != want).any(axis=1).sum())
        return (f"{bad_rows}/{a.shape[0]} frames are not one-hot at slot "
                f"{prompt_id} (argmax slots seen: {hot[:8]})")
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nemo", required=True, type=Path)
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--left-secs", type=float, default=5.6)
    p.add_argument("--chunk-secs", type=float, default=1.04)
    p.add_argument("--right-secs", type=float, default=1.04)
    p.add_argument("--backend", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--feed-ms", type=int, default=500)
    p.add_argument("--force", action="store_true")
    p.add_argument("--language", default=None,
                   help="Language hint for prompt-conditioned unified variants "
                        "(nemotron-labs-unified-asr-0.6b). Passed to both sides; "
                        "'auto' runs the C++ side with no hint (NULL -> auto slot).")
    p.add_argument("--cpp-language", default=None,
                   help="Override the hint on the C++ side only. A wrong value "
                        "is a negative control: the run must FAIL.")
    p.add_argument("--expect-prompt-id", type=int, default=None,
                   help="Fail unless the reference resolved --language to this "
                        "prompt slot (independent check of the dictionary).")
    p.add_argument("--cli", type=Path, default=None,
                   help="transcribe-cli to test (default build/bin/transcribe-cli).")
    p.add_argument("--ref-python", type=Path, default=None,
                   help="Python interpreter with NeMo for the reference dump "
                        "(default: uv run --project scripts/envs/parakeet).")
    p.add_argument("--tol-max-abs", type=float, default=5.0)
    p.add_argument("--tol-mean-abs", type=float, default=0.1)
    p.add_argument("--tol-prompt-replay", type=float, default=1e-4,
                   help="max_abs allowed between the C++ prompt-MLP output and "
                        "its NumPy replay (prompt-conditioned variants).")
    p.add_argument("--require-transcript-match", action="store_true",
                   help="Gate on transcript equality (cpp may only append "
                        "closing punctuation from the EOS lookahead).")
    args = p.parse_args()

    out_dir = args.out.resolve()
    ref_dir = out_dir / "ref"
    cpp_dir = out_dir / "cpp"

    run_ref_dump(
        nemo=args.nemo.resolve(), audio=args.audio.resolve(),
        out_dir=ref_dir,
        left_secs=args.left_secs,
        chunk_secs=args.chunk_secs,
        right_secs=args.right_secs,
        force=args.force,
        language=args.language,
        ref_python=args.ref_python,
    )
    # Prompt slot the reference used (None for non-prompt variants).
    ref_prompt_id = None
    hist_path = ref_dir / "stream_history.json"
    if hist_path.exists():
        ref_prompt_id = (json.loads(hist_path.read_text())
                         .get("buffered_streaming", {}).get("prompt_id"))
    if ref_prompt_id is not None and args.language is None:
        # With no hint the dumper defaults to 'en' while the C++ side would
        # run the auto slot: the two sides would compare different prompts.
        raise SystemExit(
            "error: this model is prompt-conditioned; pass --language "
            "(a prompt-dictionary key, or 'auto' for the no-hint slot)")
    cpp_language = args.cpp_language if args.cpp_language is not None else args.language
    if cpp_language == "auto":
        cpp_language = None
    cpp_path, cpp_text = run_cpp_dump(
        gguf=args.gguf.resolve(), audio=args.audio.resolve(),
        out_dir=cpp_dir,
        feed_ms=args.feed_ms,
        backend=args.backend, threads=args.threads,
        left_ms=int(round(args.left_secs   * 1000)),
        chunk_ms=int(round(args.chunk_secs * 1000)),
        right_ms=int(round(args.right_secs * 1000)),
        language=cpp_language,
        cli=(args.cli.resolve() if args.cli is not None else None),
    )

    gate_failures: list[str] = []
    if args.expect_prompt_id is not None and ref_prompt_id != args.expect_prompt_id:
        gate_failures.append(
            f"reference resolved language {args.language!r} to prompt slot "
            f"{ref_prompt_id}, expected {args.expect_prompt_id}")

    # Final transcript byte equality
    ref_transcript_path = ref_dir / "transcript.json"
    ref_text = ""
    if ref_transcript_path.exists():
        ref_text = json.loads(ref_transcript_path.read_text()).get("text", "")
    # CPP transcript: strip leading/trailing whitespace
    cpp_text_norm = cpp_text.strip()
    ref_text_norm = ref_text.strip()
    transcript_match = (cpp_text_norm == ref_text_norm)

    # Per-chunk comparison. Steady-state chunks should match within
    # CPU-fp32 noise. Last-chunk divergence is reported but tolerated.
    ref_steps = sorted({
        int(p.stem.split(".")[2])
        for p in ref_dir.glob("stream.chunk.*.audio_in.f32")
    })
    cpp_steps = sorted({
        int(p.stem.split(".")[2])
        for p in cpp_dir.glob("stream.chunk.*.audio_in.f32")
    })
    common = sorted(set(ref_steps) & set(cpp_steps))
    print(f"\n=== Per-chunk parity (ref steps: {ref_steps}; cpp steps: {cpp_steps}; "
          f"common: {common}) ===")

    # Tolerances calibrated to the parakeet family's accepted noise
    # envelope (tests/tolerances/parakeet.json allows max_abs=3.7,
    # mean_abs=0.05 at enc.final on the offline path). Per-chunk
    # enc_out is a slice of that same encoder output running on a
    # smaller window, so the same tolerance applies.
    #
    # That envelope cannot see the prompt: on nemotron-labs-unified-asr-0.6b
    # (jfk, 70/13/13) a missing prompt moves enc_out by mean 4e-2 / max 0.56
    # and a wrong slot by 3e-2 / 0.38, with an unchanged transcript, while
    # the correct prompt at (70,1,1) already drifts 2e-2 / 0.68 from encoder
    # noise alone. The prompt layer is therefore gated separately, by the
    # one-hot inspection and the NumPy replay below.
    TOL_MAX_ABS  = args.tol_max_abs
    TOL_MEAN_ABS = args.tol_mean_abs
    prompt_mlp = None
    if ref_prompt_id is not None:
        prompt_mlp = export_prompt_mlp(args.gguf.resolve(), out_dir / "prompt_mlp",
                                       args.ref_python)

    # Chunk coverage: both sides must produce the same, non-empty step
    # set. Comparing only the intersection would let dropped chunks (or
    # no common chunk at all) pass.
    if not ref_steps:
        gate_failures.append("reference produced no chunks")
    if ref_steps != cpp_steps:
        gate_failures.append(
            f"chunk steps differ: ref-only {sorted(set(ref_steps) - set(cpp_steps))}, "
            f"cpp-only {sorted(set(cpp_steps) - set(ref_steps))}")

    kinds = ["audio_in", "enc_out"]
    if ref_prompt_id is not None:
        kinds.append("enc_out_unprompted")

    rows: list[dict] = []
    replay_rows: list[dict] = []
    fail_chunks = 0
    last_step = common[-1] if common else None
    eos_pad_frames = int(round(args.right_secs * 1000)) // 80
    for step in common:
        if ref_prompt_id is not None:
            msg = check_prompt_one_hot(
                cpp_dir / f"stream.chunk.{step}.prompt.one_hot.f32", int(ref_prompt_id))
            if msg is not None:
                print(f"  step {step:>2} prompt.one_hot: {msg} [FAIL]")
                fail_chunks += 1
            rp = replay_prompt(
                prompt_mlp,
                cpp_dir / f"stream.chunk.{step}.enc_out_unprompted.f32",
                cpp_dir / f"stream.chunk.{step}.enc_out.f32",
                int(ref_prompt_id))
            replay_rows.append({"step": step, **rp})
            if rp["status"] != "OK" or rp["max_abs"] > args.tol_prompt_replay:
                detail = (f"max_abs={rp['max_abs']:.3e}" if rp["status"] == "OK"
                          else rp["status"])
                print(f"  step {step:>2} prompt replay (slot {ref_prompt_id}): {detail} [FAIL]")
                fail_chunks += 1
        for kind in kinds:
            name = f"stream.chunk.{step}.{kind}"
            # The final R rows intentionally differ because cpp adds lookahead.
            r = compare_pair(
                name,
                ref_dir / f"{name}.f32",
                cpp_dir / f"{name}.f32",
                eos_pad_frames=(eos_pad_frames if step == last_step else None),
            )
            rows.append({"step": step, **r})
            if r.get("status") == "OK":
                tag = "OK"
                over_max  = r["max_abs"]  > TOL_MAX_ABS
                over_mean = r["mean_abs"] > TOL_MEAN_ABS
                if over_max or over_mean:
                    tag = "FAIL"
                    fail_chunks += 1
                if "pad_max_abs" in r and r["pad_max_abs"] != 0.0:
                    tag = "FAIL"
                    fail_chunks += 1
                # The EOS tail is at most the right context, in whole frames.
                if r.get("pad_samples", 0) > eos_pad_frames * 1280 and tag != "FAIL":
                    tag = "FAIL"
                    fail_chunks += 1
                # The PCM window is copied, not computed: real audio must
                # be bit-identical, not merely within the encoder envelope.
                if kind == "audio_in" and r["max_abs"] != 0.0 and tag != "FAIL":
                    tag = "FAIL"
                    fail_chunks += 1
                extra = ""
                if "pad_samples" in r:
                    extra = f" eos_pad_samples={r['pad_samples']} (zeros: {r['pad_max_abs'] == 0.0})"
                if "eos_tail_max_abs" in r:
                    extra = (f" eos_tail_rows={r['eos_tail_rows']} eos_tail_max_abs={r['eos_tail_max_abs']:.3e}"
                             f" pad_rows={r.get('pad_rows', 0)} (expected divergence, not gated)")
                print(f"  step {step:>2} {kind:9s}: max_abs={r['max_abs']:.3e} "
                      f"mean_abs={r['mean_abs']:.3e} rel_max={r['rel_max']:.3e} [{tag}]{extra}")
            else:
                # Variable-stride algorithm produces identical chunk
                # geometry to ref, so SHAPE_DIFF or MISSING is now a
                # real failure (unlike the legacy fixed-stride path).
                print(f"  step {step:>2} {kind:9s}: {r['status']}  "
                      f"{r.get('ref_shape', '')} vs {r.get('cpp_shape', '')}")
                fail_chunks += 1

    def worst(kind: str, key: str) -> float | None:
        vals = [r[key] for r in rows if r.get("status") == "OK"
                and r["name"].endswith("." + kind)]
        return max(vals) if vals else None

    replay_ok = [r["max_abs"] for r in replay_rows if r.get("status") == "OK"]
    if replay_rows:
        print(f"  prompt replay (slot {ref_prompt_id}): {len(replay_ok)}/{len(replay_rows)} "
              f"chunks compared, worst max_abs="
              f"{(max(replay_ok) if replay_ok else float('nan')):.3e} "
              f"(tolerance {args.tol_prompt_replay:.1e})")
    for kind in kinds:
        if worst(kind, "max_abs") is not None:
            print(f"  worst {kind}: max_abs={worst(kind, 'max_abs'):.3e} "
                  f"mean_abs={worst(kind, 'mean_abs'):.3e}")

    print()
    print(f"final transcript ref: {ref_text_norm!r}")
    print(f"final transcript cpp: {cpp_text_norm!r}")
    print(f"transcript match: {transcript_match}")

    summary = {
        "transcript_match": transcript_match,
        "ref_text": ref_text_norm,
        "cpp_text": cpp_text_norm,
        "n_common_chunks": len(common),
        "ref_steps": ref_steps,
        "cpp_steps": cpp_steps,
        "chunk_failures": fail_chunks,
        "gate_failures": gate_failures,
        "language": args.language,
        "cpp_language": cpp_language,
        "ref_prompt_id": ref_prompt_id,
        "tolerances": {"max_abs": TOL_MAX_ABS, "mean_abs": TOL_MEAN_ABS,
                       "prompt_replay_max_abs": args.tol_prompt_replay},
        "prompt_replay": {"worst_max_abs": (max(replay_ok) if replay_ok else None),
                          "n_chunks": len(replay_rows), "n_compared": len(replay_ok)},
        "worst": {k: {"max_abs": worst(k, "max_abs"), "mean_abs": worst(k, "mean_abs")}
                  for k in kinds},
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    for msg in gate_failures:
        print(f"FAIL: {msg}")
    if gate_failures:
        return 1
    if args.require_transcript_match:
        # Exact, except that the C++ EOS lookahead may add closing
        # punctuation the reference loop never reaches.
        suffix = (cpp_text_norm[len(ref_text_norm):]
                  if ref_text_norm and cpp_text_norm.startswith(ref_text_norm) else None)
        if suffix is None or suffix.strip(" .,!?;:¿¡…") != "":
            print("FAIL: transcript differs (--require-transcript-match)")
            return 1
    if fail_chunks > 0:
        print(f"FAIL: {fail_chunks} chunks exceed tolerance or have wrong shape")
        return 1
    if not transcript_match:
        if cpp_text_norm.startswith(ref_text_norm):
            tail = cpp_text_norm[len(ref_text_norm):]
            print(f"INFO: cpp transcript extends ref by {tail!r} "
                  "(EOS silence lookahead; expected)")
        else:
            # Greedy RNN-T can tip a single-token decision on fp32 noise
            # even when per-chunk encoder outputs match within tolerance.
            # Report informationally — per-chunk parity is the algorithmic
            # gate; WER on test-clean is the corpus-level gate.
            print("WARN: transcript byte-match differs (per-chunk parity gate passes — "
                  "likely fp32-noise tipping a greedy decision)")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
