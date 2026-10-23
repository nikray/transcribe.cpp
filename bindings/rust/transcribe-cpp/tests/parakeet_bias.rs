//! Generic Parakeet vocabulary input and optional boost tuning.
//!
//! Offline: TRANSCRIBE_SMOKE_PARAKEET_MODEL + TRANSCRIBE_BIAS_AUDIO.
//! Streaming: TRANSCRIBE_BIAS_STREAM_MODEL + TRANSCRIBE_BIAS_AUDIO (either
//! cache-aware Nemotron or buffered Parakeet). Audio must be 16 kHz mono WAV.
//! TRANSCRIBE_BIAS_WORDS is a comma-separated list. Optional
//! TRANSCRIBE_BIAS_EXPECT_WORD / TRANSCRIBE_BIAS_EXPECT_STREAM_WORD require a
//! genuine recognition flip: the plain output lacks the spelling and the
//! biased output contains it. No model downloads are performed by these tests.

mod common;

use transcribe_cpp::sys::{
    TRANSCRIBE_EXT_KIND_PARAKEET_BIAS_RUN, TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM,
    TRANSCRIBE_EXT_KIND_PARAKEET_STREAM,
};
use transcribe_cpp::{
    ExtSlot, Feature, Model, ParakeetBiasOptions, ParakeetBufferedStreamOptions,
    ParakeetStreamOptions, RunExtension, RunOptions, Session, StreamExtension, StreamOptions,
    Transcript,
};

fn words() -> Vec<String> {
    std::env::var("TRANSCRIBE_BIAS_WORDS")
        .unwrap_or_else(|_| "ExpertChain,Quellium".to_string())
        .split(',')
        .map(str::to_string)
        .collect()
}

fn assert_same_output(expected: &Transcript, actual: &Transcript) {
    // Only wall-clock measurements are excluded: text, raw text, token/word
    // rows, timestamps, confidence values and language must all agree.
    let mut expected = expected.clone();
    let mut actual = actual.clone();
    expected.timings = Default::default();
    actual.timings = Default::default();
    assert_eq!(expected, actual);
}

fn assert_flip(plain: &Transcript, biased: &Transcript, variable: &str) {
    eprintln!("plain: {:?}\nbiased: {:?}", plain.text, biased.text);
    if let Ok(expect) = std::env::var(variable) {
        assert!(!expect.is_empty(), "{variable} must not be empty");
        assert!(
            !plain.text.contains(&expect),
            "fixture does not demonstrate a flip: plain output already contains {expect:?}: {:?}",
            plain.text
        );
        assert!(
            biased.text.contains(&expect),
            "biased output should contain {expect:?}; got {:?} (plain: {:?})",
            biased.text,
            plain.text
        );
    }
}

#[test]
fn parakeet_bias_run_extension() {
    let (Some(path), Some(pcm)) = (
        common::family_model_pub("TRANSCRIBE_SMOKE_PARAKEET_MODEL"),
        common::audio_from_env("TRANSCRIBE_BIAS_AUDIO"),
    ) else {
        eprintln!("skip: set TRANSCRIBE_SMOKE_PARAKEET_MODEL and TRANSCRIBE_BIAS_AUDIO");
        return;
    };
    let model = Model::load(path).unwrap();
    assert!(model.supports(Feature::Vocabulary));
    assert!(model.accepts_ext(ExtSlot::Run, TRANSCRIBE_EXT_KIND_PARAKEET_BIAS_RUN));
    let mut session = model.session().unwrap();
    let plain = session.run(&pcm, &RunOptions::default()).unwrap();
    assert!(!plain.text.trim().is_empty());

    let empty = RunOptions {
        vocabulary: Vec::new(),
        family: Some(RunExtension::ParakeetBias(ParakeetBiasOptions::default())),
        ..Default::default()
    };
    assert_same_output(&plain, &session.run(&pcm, &empty).unwrap());

    let generic = RunOptions {
        vocabulary: words(),
        ..Default::default()
    };
    let biased = session.run(&pcm, &generic).unwrap();
    assert!(!biased.text.trim().is_empty());
    assert_flip(&plain, &biased, "TRANSCRIBE_BIAS_EXPECT_WORD");

    let explicit_default = RunOptions {
        family: Some(RunExtension::ParakeetBias(ParakeetBiasOptions::default())),
        ..generic.clone()
    };
    assert_same_output(&biased, &session.run(&pcm, &explicit_default).unwrap());
    let disabled = RunOptions {
        family: Some(RunExtension::ParakeetBias(ParakeetBiasOptions {
            boost: Some(0.0),
        })),
        ..generic
    };
    assert_same_output(&plain, &session.run(&pcm, &disabled).unwrap());
    assert_same_output(&plain, &session.run(&pcm, &RunOptions::default()).unwrap());
}

fn stream_options(model: &Model, boost: Option<f32>) -> StreamOptions {
    let bias = Some(ParakeetBiasOptions { boost });
    let family = if model.accepts_ext(ExtSlot::Stream, TRANSCRIBE_EXT_KIND_PARAKEET_STREAM) {
        StreamExtension::ParakeetStream(ParakeetStreamOptions {
            bias,
            ..Default::default()
        })
    } else {
        assert!(model.accepts_ext(
            ExtSlot::Stream,
            TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM
        ));
        StreamExtension::ParakeetBuffered(ParakeetBufferedStreamOptions {
            bias,
            ..Default::default()
        })
    };
    StreamOptions {
        family: Some(family),
        ..Default::default()
    }
}

fn decode_stream(
    session: &mut Session,
    run: RunOptions,
    options: StreamOptions,
    pcm: &[f32],
) -> Transcript {
    let mut stream = session.stream(&run, &options).unwrap();
    // Neither source options nor the Rust FFI keepalives outlive begin. The
    // native session must own everything needed by subsequent feeds.
    drop(run);
    drop(options);
    for chunk in pcm.chunks(997) {
        stream.feed(chunk).unwrap();
    }
    assert!(stream.finalize().unwrap().is_final);
    stream.snapshot()
}

#[test]
fn parakeet_generic_vocabulary_streaming() {
    let (Some(path), Some(pcm)) = (
        common::family_model_pub("TRANSCRIBE_BIAS_STREAM_MODEL"),
        common::audio_from_env("TRANSCRIBE_BIAS_AUDIO"),
    ) else {
        eprintln!("skip: set TRANSCRIBE_BIAS_STREAM_MODEL and TRANSCRIBE_BIAS_AUDIO");
        return;
    };
    let model = Model::load(path).unwrap();
    assert!(model.supports(Feature::Vocabulary));
    let mut session = model.session().unwrap();
    let plain = decode_stream(
        &mut session,
        RunOptions::default(),
        StreamOptions::default(),
        &pcm,
    );
    assert!(!plain.text.trim().is_empty());
    let empty = decode_stream(
        &mut session,
        RunOptions::default(),
        stream_options(&model, None),
        &pcm,
    );
    assert_same_output(&plain, &empty);
    let generic = RunOptions {
        vocabulary: words(),
        ..Default::default()
    };
    let biased = decode_stream(
        &mut session,
        generic.clone(),
        StreamOptions::default(),
        &pcm,
    );
    assert!(!biased.text.trim().is_empty());
    assert_flip(&plain, &biased, "TRANSCRIBE_BIAS_EXPECT_STREAM_WORD");
    let explicit_default = decode_stream(
        &mut session,
        generic.clone(),
        stream_options(&model, None),
        &pcm,
    );
    assert_same_output(&biased, &explicit_default);
    let disabled = decode_stream(
        &mut session,
        generic.clone(),
        stream_options(&model, Some(0.0)),
        &pcm,
    );
    assert_same_output(&plain, &disabled);

    // Abandon an active biased stream after feeding part of the clip. Reset
    // and the next stream must discard both partial matching and vocabulary.
    {
        let mut stream = session.stream(&generic, &StreamOptions::default()).unwrap();
        stream.feed(&pcm[..pcm.len().min(16000)]).unwrap();
        stream.reset();
        assert!(stream.text().full.is_empty());
    }
    let after_reset = decode_stream(
        &mut session,
        RunOptions::default(),
        StreamOptions::default(),
        &pcm,
    );
    assert_same_output(&plain, &after_reset);
}
