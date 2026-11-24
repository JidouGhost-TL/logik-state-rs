//! Byte-exact `libmgba.so.0.10` ARM7TDMI Golden Integration Tests for `logik-state`.
//!
//! Verifies all **15** clean-room ARM7TDMI execution golden vectors (**1,856 / 1,856 bytes**,
//! strict `assert_eq!` with zero tolerance) covering all **6** Logik State GBAModPlay engine
//! revisions discovered across the 3,374-ROM GBA corpus:
//!
//! 1. `LogikVersion::V1Mod1` (2 vectors, 256 bytes):
//!    - Fractional resampled looped voice (`lsr #12`, `and #0xFFF` wrap, `lsr #8`)
//!    - 2-voice wrapping 8-bit sum
//! 2. `LogikVersion::V2Mod2Halfword` (3 vectors, 320 bytes):
//!    - General looped voice (16-bit halfword length/loop)
//!    - `vol == 64` (`asr #2`) fast path + loop wrap
//!    - `len == 2` early-exit guard
//! 3. `LogikVersion::V2Mod2Word` (2 vectors, 256 bytes):
//!    - General looped voice (32-bit word length/loop)
//!    - `vol == 64` (`asr #2`) fast path + loop wrap
//! 4. `LogikVersion::V3EarlyVol64` (3 vectors, 384 bytes):
//!    - Q20.12 general looped voice (`sub r4, r4, sl` phase preservation)
//!    - Q20.12 one-shot voice (`tst sl, sl; beq` early stop)
//!    - `vol == 64` fast path exercising the authentic post-loop stale-`r6` hardware quirk
//! 5. `LogikVersion::V3EarlyVol64Adpcm` (3 vectors, 384 bytes):
//!    - Single 4-bit IMA ADPCM voice (high-nibble-first, `((nib & 7) * step) >> 2`, 2x doubled)
//!    - Mixed PCM8 + 4-bit IMA ADPCM 2-voice wrapping sum
//!    - Odd-count (`63 + 65`) ADPCM resume verifying exact state persistence across calls
//! 6. `LogikVersion::V3Standard` (2 vectors, 256 bytes):
//!    - Q20.12 resampled looped voice at `vol == 64` (clean loop, no stale-`r6` quirk)
//!    - 2-voice looped + one-shot wrapping 8-bit sum

use logik_state::{
    AccuracySynth, LogikAdpcmVoice, LogikPcmVoice, LogikSampleView, LogikVersion, StudioSynth,
};

const MANIFEST_JSON: &str = include_str!("golden/manifest.json");

const GOLDEN_V1_RESAMPLED_LOOP: &[u8] = include_bytes!("golden/v1_mod1_resampled_loop.bin");
const GOLDEN_V1_2VOICE_WRAP: &[u8] = include_bytes!("golden/v1_mod1_2voice_wrap.bin");

const GOLDEN_V2_HW_GENERAL_LOOP: &[u8] = include_bytes!("golden/v2_halfword_general_loop.bin");
const GOLDEN_V2_HW_VOL64_FAST: &[u8] = include_bytes!("golden/v2_halfword_vol64_fast.bin");
const GOLDEN_V2_HW_LEN2_GUARD: &[u8] = include_bytes!("golden/v2_halfword_len2_guard.bin");

const GOLDEN_V2_W_GENERAL_LOOP: &[u8] = include_bytes!("golden/v2_word_general_loop.bin");
const GOLDEN_V2_W_VOL64_FAST: &[u8] = include_bytes!("golden/v2_word_vol64_fast.bin");

const GOLDEN_V3_EARLY_GENERAL_LOOP: &[u8] = include_bytes!("golden/v3_early_general_loop.bin");
const GOLDEN_V3_EARLY_ONESHOT: &[u8] = include_bytes!("golden/v3_early_oneshot.bin");
const GOLDEN_V3_EARLY_VOL64_QUIRK: &[u8] = include_bytes!("golden/v3_early_vol64_quirk.bin");

const GOLDEN_V3_ADPCM_SINGLE: &[u8] = include_bytes!("golden/v3_adpcm_single_voice.bin");
const GOLDEN_V3_ADPCM_2VOICE: &[u8] = include_bytes!("golden/v3_adpcm_2voice_mix.bin");
const GOLDEN_V3_ADPCM_ODD_RESUME: &[u8] = include_bytes!("golden/v3_adpcm_odd_resume.bin");

const GOLDEN_V3_STD_RESAMPLED_LOOP: &[u8] =
    include_bytes!("golden/v3_standard_resampled_loop.bin");
const GOLDEN_V3_STD_2VOICE_WRAP: &[u8] = include_bytes!("golden/v3_standard_2voice_wrap.bin");

/// Clean-room deterministic synthetic PCM8 waveform A (identical to the `libmgba` ARM harness).
fn synth_wave_a() -> [u8; 256] {
    let mut out = [0u8; 256];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut v = (((i as i32) * 19 + 7) % 256) - 128;
        if i % 5 == 0 {
            v = -v;
        }
        v = v.clamp(-128, 127);
        *slot = v as i8 as u8;
    }
    out
}

/// Clean-room deterministic synthetic PCM8 waveform B (identical to the `libmgba` ARM harness).
fn synth_wave_b() -> [u8; 256] {
    let mut out = [0u8; 256];
    for (i, slot) in out.iter_mut().enumerate() {
        let v = (((i as i32) * 37 + 53) % 220) - 110;
        *slot = v as i8 as u8;
    }
    out
}

/// Clean-room deterministic synthetic 4-bit IMA ADPCM stream A (identical to the `libmgba` harness).
fn synth_adpcm_a() -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = ((i * 5 + 3) & 0x0F) as u8;
        let lo = ((i * 11 + 7) & 0x0F) as u8;
        *slot = (hi << 4) | lo;
    }
    out
}

fn assert_exact_bytes(actual: &[i8], expected: &[u8], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e as i8, "{label}: byte mismatch at index {i}");
    }
}

#[test]
fn golden_manifest_is_present_and_complete() {
    assert!(MANIFEST_JSON.contains("Logik State GBAModPlay"));
    assert!(MANIFEST_JSON.contains("\"total_bytes\": 1856"));
}

#[test]
fn golden_v1_mod1_resampled_loop() {
    let wave_a = synth_wave_a();
    let mut voice = LogikPcmVoice::new_v1_v2(0, 0, 0x1345, 64, 16, 48, 48);
    let mut out = [0i8; 128];
    AccuracySynth::mix_pcm_voice(LogikVersion::V1Mod1, &wave_a, &mut voice, &mut out);
    assert_exact_bytes(&out, GOLDEN_V1_RESAMPLED_LOOP, "v1_mod1_resampled_loop");
}

#[test]
fn golden_v1_mod1_2voice_wrap() {
    let wave_a = synth_wave_a();
    let wave_b = synth_wave_b();
    let mut v1 = LogikPcmVoice::new_v1_v2(0, 0, 0x1000, 128, 0, 128, 64);
    let mut v2 = LogikPcmVoice::new_v1_v2(0, 0x0800, 0x1555, 64, 8, 56, 56);
    let mut out = [0i8; 128];
    AccuracySynth::mix_v1_mod1(&wave_a, &mut v1, &mut out);
    AccuracySynth::mix_v1_mod1(&wave_b, &mut v2, &mut out);
    assert_exact_bytes(&out, GOLDEN_V1_2VOICE_WRAP, "v1_mod1_2voice_wrap");
}

#[test]
fn golden_v2_halfword_general_loop() {
    let wave_a = synth_wave_a();
    let mut voice = LogikPcmVoice::new_v1_v2(0, 0, 0x1567, 64, 16, 48, 52);
    let mut out = [0i8; 128];
    AccuracySynth::mix_pcm_voice(LogikVersion::V2Mod2Halfword, &wave_a, &mut voice, &mut out);
    assert_exact_bytes(&out, GOLDEN_V2_HW_GENERAL_LOOP, "v2_halfword_general_loop");
}

#[test]
fn golden_v2_halfword_vol64_fast() {
    let wave_a = synth_wave_a();
    let mut voice = LogikPcmVoice::new_v1_v2(0, 0, 0x1345, 48, 12, 36, 64);
    let mut out = [0i8; 128];
    AccuracySynth::mix_v2_halfword(&wave_a, &mut voice, &mut out);
    assert_exact_bytes(&out, GOLDEN_V2_HW_VOL64_FAST, "v2_halfword_vol64_fast");
}

#[test]
fn golden_v2_halfword_len2_guard() {
    let wave_a = synth_wave_a();
    let mut voice = LogikPcmVoice::new_v1_v2(0, 0, 0x1000, 2, 0, 2, 64);
    let mut out = [0i8; 64];
    AccuracySynth::mix_v2_halfword(&wave_a, &mut voice, &mut out);
    assert_exact_bytes(&out, GOLDEN_V2_HW_LEN2_GUARD, "v2_halfword_len2_guard");
}

#[test]
fn golden_v2_word_general_loop() {
    let wave_a = synth_wave_a();
    let mut voice = LogikPcmVoice::new_v1_v2(0, 0, 0x1789, 64, 16, 48, 45);
    let mut out = [0i8; 128];
    AccuracySynth::mix_pcm_voice(LogikVersion::V2Mod2Word, &wave_a, &mut voice, &mut out);
    assert_exact_bytes(&out, GOLDEN_V2_W_GENERAL_LOOP, "v2_word_general_loop");
}

#[test]
fn golden_v2_word_vol64_fast() {
    let wave_b = synth_wave_b();
    let mut voice = LogikPcmVoice::new_v1_v2(0, 0, 0x1234, 48, 8, 40, 64);
    let mut out = [0i8; 128];
    AccuracySynth::mix_v2_word(&wave_b, &mut voice, &mut out);
    assert_exact_bytes(&out, GOLDEN_V2_W_VOL64_FAST, "v2_word_vol64_fast");
}

#[test]
fn golden_v3_early_general_loop() {
    let wave_a = synth_wave_a();
    let mut voice = LogikPcmVoice::new_v3(0, 0, 0x1654, 64, 16, 48, 50);
    let mut out = [0i8; 128];
    AccuracySynth::mix_pcm_voice(LogikVersion::V3EarlyVol64, &wave_a, &mut voice, &mut out);
    assert_exact_bytes(&out, GOLDEN_V3_EARLY_GENERAL_LOOP, "v3_early_general_loop");
}

#[test]
fn golden_v3_early_oneshot() {
    let wave_a = synth_wave_a();
    let mut voice = LogikPcmVoice::new_v3(0, 0, 0x1400, 64, 0, 0, 58);
    let mut out = [0i8; 128];
    AccuracySynth::mix_v3_early_vol64(&wave_a, &mut voice, &mut out);
    assert!(!voice.active);
    assert_exact_bytes(&out, GOLDEN_V3_EARLY_ONESHOT, "v3_early_oneshot");
}

#[test]
fn golden_v3_early_vol64_quirk() {
    let wave_a = synth_wave_a();
    let mut voice = LogikPcmVoice::new_v3(0, 0, 0x1654, 48, 12, 36, 64);
    let mut out = [0i8; 128];
    AccuracySynth::mix_v3_early_vol64(&wave_a, &mut voice, &mut out);
    assert_exact_bytes(&out, GOLDEN_V3_EARLY_VOL64_QUIRK, "v3_early_vol64_quirk");
}

#[test]
fn golden_v3_adpcm_single_voice() {
    let adpcm_a = synth_adpcm_a();
    let mut voice = LogikAdpcmVoice::new(0, 64, 64);
    let mut out = [0i8; 128];
    AccuracySynth::mix_v3_adpcm_voice(&adpcm_a, &mut voice, &mut out);
    assert_exact_bytes(&out, GOLDEN_V3_ADPCM_SINGLE, "v3_adpcm_single_voice");
}

#[test]
fn golden_v3_adpcm_2voice_mix() {
    let wave_a = synth_wave_a();
    let adpcm_a = synth_adpcm_a();
    let mut pcm_v = LogikPcmVoice::new_v3(0, 0, 0x1200, 64, 16, 48, 40);
    let mut ima_v = LogikAdpcmVoice::new(0, 64, 48);
    let mut out = [0i8; 128];
    AccuracySynth::mix_pcm_voice(
        LogikVersion::V3EarlyVol64Adpcm,
        &wave_a,
        &mut pcm_v,
        &mut out,
    );
    AccuracySynth::mix_v3_adpcm_voice(&adpcm_a, &mut ima_v, &mut out);
    assert_exact_bytes(&out, GOLDEN_V3_ADPCM_2VOICE, "v3_adpcm_2voice_mix");
}

#[test]
fn golden_v3_adpcm_odd_resume() {
    let adpcm_a = synth_adpcm_a();
    let mut voice = LogikAdpcmVoice::new(0, 64, 56);
    let mut out = [0i8; 128];
    AccuracySynth::mix_v3_adpcm_voice(&adpcm_a, &mut voice, &mut out[..63]);
    AccuracySynth::mix_v3_adpcm_voice(&adpcm_a, &mut voice, &mut out[63..]);
    assert_exact_bytes(&out, GOLDEN_V3_ADPCM_ODD_RESUME, "v3_adpcm_odd_resume");
}

#[test]
fn golden_v3_standard_resampled_loop() {
    let wave_a = synth_wave_a();
    let mut voice = LogikPcmVoice::new_v3(0, 0, 0x1432, 64, 16, 48, 64);
    let mut out = [0i8; 128];
    AccuracySynth::mix_pcm_voice(LogikVersion::V3Standard, &wave_a, &mut voice, &mut out);
    assert_exact_bytes(
        &out,
        GOLDEN_V3_STD_RESAMPLED_LOOP,
        "v3_standard_resampled_loop",
    );
}

#[test]
fn golden_v3_standard_2voice_wrap() {
    let wave_a = synth_wave_a();
    let wave_b = synth_wave_b();
    let mut v1 = LogikPcmVoice::new_v3(0, 0, 0x1432, 64, 16, 48, 55);
    let mut v2 = LogikPcmVoice::new_v3(0, 0x0400, 0x1800, 64, 0, 0, 60);
    let mut out = [0i8; 128];
    AccuracySynth::mix_v3_standard(&wave_a, &mut v1, &mut out);
    AccuracySynth::mix_v3_standard(&wave_b, &mut v2, &mut out);
    assert_exact_bytes(&out, GOLDEN_V3_STD_2VOICE_WRAP, "v3_standard_2voice_wrap");
}

#[test]
fn studio_synth_renders_looped_and_oneshot() {
    let wave_a = synth_wave_a();
    let view = LogikSampleView::from_pcm_bytes(10_512, 64, Some(16), Some(64), &wave_a);
    let studio = StudioSynth::default();
    let mut out_l = [0.0f32; 256];
    let mut out_r = [0.0f32; 256];
    let mut pos = 0.0;
    studio.render_voice_f32(&view, &mut pos, 10_512.0, 0.85, 0.25, &mut out_l, &mut out_r);
    assert!(out_l.iter().any(|&v| v.abs() > 0.01));
    assert!(out_r.iter().any(|&v| v.abs() > 0.01));
    assert!(pos > 50.0);
}
