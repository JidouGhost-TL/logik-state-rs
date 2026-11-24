use std::hint::black_box;
use std::time::Instant;

use logik_state::{
    AccuracySynth, LogikAdpcmVoice, LogikPcmVoice, LogikSampleView, LogikVersion, StudioSynth,
};

fn main() {
    println!("=== logik-state-rs Audio Performance Benchmarks ===");

    let pcm: Vec<u8> = (0i32..4096)
        .map(|i| (((i * 19 + 7) % 256) - 128) as i8 as u8)
        .collect();
    let adpcm: Vec<u8> = (0..1024)
        .map(|i| ((((i * 5 + 3) & 0x0F) << 4) | ((i * 11 + 7) & 0x0F)) as u8)
        .collect();
    let sample = LogikSampleView::from_pcm_bytes(10_512, 64, Some(512), Some(4096), &pcm);

    let mut out_128 = [0i8; 128];
    let iters = 300_000usize;

    // 1. AccuracySynth V3Standard (128 frames)
    let start = Instant::now();
    for _ in 0..iters {
        let mut voice = LogikPcmVoice::new_v3(0, 0, 0x1432, 4096, 512, 3584, 55);
        AccuracySynth::mix_pcm_voice(
            LogikVersion::V3Standard,
            black_box(&pcm),
            &mut voice,
            &mut out_128,
        );
        black_box(&out_128);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::v3_standard (128f):                  {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 128) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 2. AccuracySynth V2Mod2Halfword vol==64 fast path (128 frames)
    let start = Instant::now();
    for _ in 0..iters {
        let mut voice = LogikPcmVoice::new_v1_v2(0, 0, 0x1345, 4096, 512, 3584, 64);
        AccuracySynth::mix_v2_halfword(black_box(&pcm), &mut voice, &mut out_128);
        black_box(&out_128);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::v2_halfword_vol64 (128f):            {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 128) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 3. AccuracySynth V3EarlyVol64Adpcm 4-bit IMA ADPCM (128 frames)
    let start = Instant::now();
    for _ in 0..iters {
        let mut voice = LogikAdpcmVoice::new(0, 64, 64);
        AccuracySynth::mix_v3_adpcm_voice(black_box(&adpcm), &mut voice, &mut out_128);
        black_box(&out_128);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::v3_adpcm_2x (128f):                  {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 128) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 4. StudioSynth 48 kHz Lanczos-3 (1024 frames)
    let studio = StudioSynth::default();
    let mut out_l = vec![0.0f32; 1024];
    let mut out_r = vec![0.0f32; 1024];
    let studio_iters = 50_000usize;
    let start = Instant::now();
    for _ in 0..studio_iters {
        let mut pos = 0.0f64;
        studio.render_voice_f32(
            black_box(&sample),
            &mut pos,
            10_512.0,
            0.85,
            0.25,
            &mut out_l,
            &mut out_r,
        );
        black_box((&out_l, &out_r));
    }
    let elapsed = start.elapsed();
    println!(
        "StudioSynth::render_voice_f32 (48kHz Lanczos3, 1k): {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / studio_iters as f64,
        ((studio_iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );
}
