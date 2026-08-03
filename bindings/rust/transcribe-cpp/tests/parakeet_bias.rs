//! Parakeet custom-vocabulary biasing (run-slot PKBR extension).
//!
//! Model-gated: resolves an offline parakeet GGUF from
//! `TRANSCRIBE_SMOKE_PARAKEET_MODEL` and bias audio from
//! `TRANSCRIBE_BIAS_AUDIO` (16 kHz mono wav), skipping cleanly when
//! absent. When `TRANSCRIBE_BIAS_EXPECT_WORD` is also set, the biased
//! transcript must contain it — the flip assertion used with a fixture
//! clip containing an invented noun the unbiased decode near-misses.

mod common;

use transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_PARAKEET_BIAS_RUN;
use transcribe_cpp::{ExtSlot, Model, ParakeetBiasOptions, RunExtension, RunOptions};

#[test]
fn parakeet_bias_run_extension() {
    let model_path = match common::family_model_pub("TRANSCRIBE_SMOKE_PARAKEET_MODEL") {
        Some(p) => p,
        None => {
            eprintln!("skip parakeet_bias_run_extension: set TRANSCRIBE_SMOKE_PARAKEET_MODEL");
            return;
        }
    };
    let pcm = match common::audio_from_env("TRANSCRIBE_BIAS_AUDIO") {
        Some(a) => a,
        None => {
            eprintln!("skip parakeet_bias_run_extension: set TRANSCRIBE_BIAS_AUDIO");
            return;
        }
    };

    let model = Model::load(&model_path).unwrap();
    assert!(
        model.accepts_ext(ExtSlot::Run, TRANSCRIBE_EXT_KIND_PARAKEET_BIAS_RUN),
        "parakeet model should accept the PKBR run extension"
    );
    let mut session = model.session().unwrap();

    let plain = session.run(&pcm, &RunOptions::default()).unwrap();
    assert!(!plain.text.trim().is_empty());

    let words: Vec<String> = std::env::var("TRANSCRIBE_BIAS_WORDS")
        .unwrap_or_else(|_| "ExpertChain,Quellium".to_string())
        .split(',')
        .map(str::to_string)
        .collect();
    let opts = RunOptions {
        family: Some(RunExtension::ParakeetBias(ParakeetBiasOptions {
            words,
            boost: None, // library default (3.0)
        })),
        ..Default::default()
    };
    let biased = session.run(&pcm, &opts).unwrap();
    assert!(!biased.text.trim().is_empty());

    if let Ok(expect) = std::env::var("TRANSCRIBE_BIAS_EXPECT_WORD") {
        assert!(
            biased.text.contains(&expect),
            "biased transcript should contain {expect:?}; got {:?} (unbiased: {:?})",
            biased.text,
            plain.text
        );
    }
}
