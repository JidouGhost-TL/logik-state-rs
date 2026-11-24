# logik-state-rs

**Zero-dependency, pure Rust parser, software synthesizer, and byte-exact golden-tested emulator for the Logik State GBAModPlay GBA Sound Engine (Game Boy Advance).**

[![Crates.io](https://img.shields.io/crates/v/logik-state.svg)](https://crates.io/crates/logik-state)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](#features)

**Logik State GBAModPlay** powered **60 commercial Game Boy Advance releases** across **6 distinct ARM7TDMI mixer revisions** — from *Pinball Challenge Deluxe* (`GBAMOD1.`), *Aero the Acro-Bat* and *Zapper* (`GBAMOD2.`), *Stuntman* and *Super Puzzle Fighter II* (`GBAMOD30` + 4-bit IMA ADPCM), through 47 v3.0b/v3.1 releases including *Need for Speed: Porsche Unleashed / Underground 2 / Most Wanted / Carbon*, *Medal of Honor: Infiltrator*, *Driv3r*, *Asterix & Obelix XXL*, *Alien Hominid*, *Totally Spies! 1 & 2*, *Bionicle: Maze of Shadows*, *Lego Knights' Kingdom*, *Garfield*, *Duel Masters*, and *Hot Wheels: Stunt Track Challenge*.

`logik-state` provides a clean-room, reverse-engineered implementation designed for audio preservation, DAWs, PC source ports, and emulators.

> **Looking for a ready-to-use CLI?** You can scan GBA ROMs, extract WAV samples, and compile SoundFont 2.04 (.sf2) banks directly from the command line using our **[gbakit](https://github.com/JidouGhost-TL/gbakit)** toolkit (`gbakit audio scan`, `gbakit audio export`, `gbakit audio soundfont`).

---

## Features

- **Zero External Dependencies**: Pure Rust with no dependencies in `[dependencies]`, `#![deny(unsafe_code)]`.
- **`#![no_std]` Compatible**: Ready for embedded targets and WASM runtimes (`alloc` optional).
- **Native Module & Sample Parsers**: Zero-allocation parsers for `GBAMOD1.` / `GBAMOD2.` / `GBAMOD30` song headers (`LogikModHeader`), 20-byte V2 descriptors (`LogikSampleDescV2`), and 24-byte V3 descriptors (`LogikSampleDescV3`).
- **Dual Synthesis Engines**:
  - **`AccuracySynth`**: 100% bit-exact emulation of all **6** Logik State GBAModPlay ARM7TDMI mixer revisions — including 16-bit & 32-bit integer-index v1/v2 mixers, Q20.12 v3 mixers, the authentic `V3EarlyVol64` post-loop stale-`r6` hardware quirk, and the dedicated 2x-upsampled high-nibble-first 4-bit IMA ADPCM ARM mixer (`Super Puzzle Fighter II`).
  - **`StudioSynth`**: Renders at 48,000 Hz, 32-bit floating-point precision with a compile-time 1,024-phase Lanczos-3 polyphase windowed-sinc LUT (plus general Lanczos-`a` fallback) and true equal-power stereo panning.
- **Byte-Exact `libmgba` ARM Golden Tested**: **1,856 / 1,856** output bytes verified strictly byte-for-byte (`assert_eq!`, zero tolerance) across all 6 engine revisions.

---

## Supported Logik State GBAModPlay Revisions

Auditing all 3,374 GBA ROMs identifies **6 distinct ARM7TDMI mixer revisions** spanning **60 commercial titles**, all detected via `LogikVersion::detect_from_bytes` and byte-exact golden-tested:

| Revision | Corpus Share | Representative Titles | Song Magic | Mixer ARM7TDMI Details | IMA ADPCM |
| --- | ---: | --- | --- | --- | --- |
| `V1Mod1` | 1 ROM | *Pinball Challenge Deluxe* (`APLP`) | `GBAMOD1.` | 16-bit halfword len/loop, unsigned `lsr #12`, `and #0xFFF` wrap, `lsr #8`, no `len == 2` guard | No |
| `V2Mod2Halfword` | 4 ROMs | *Aero the Acro-Bat* (`AAOE`/`AAOJ`/`AAOP`) | `GBAMOD2.` | 16-bit halfword len/loop, `cmp sl, #2` guard, `vol == 64` (`asr #2`) fast path, `and #0xFFF` wrap | No |
| `V2Mod2Word` | 2 ROMs | *Zapper: One Wicked Cricket!* (`AZPE`/`AZPP`) | `GBAMOD2.` | 32-bit word len/loop, `cmp sl, #2` guard, `vol == 64` fast path, `and #0xFFF` wrap | No |
| `V3EarlyVol64` | 3 ROMs | *Stuntman* (`AUXE`/`AUXP`), *2-in-1: V-Rally 3 + Stuntman* (`BCSP`) | `GBAMOD30` | Q20.12 `cmp r4, sl`, `tst sl, sl` guard, `sub r4, r4, sl` wrap, `vol == 64` fast path (post-loop stale-`r6` quirk) | No |
| `V3EarlyVol64Adpcm` | 3 ROMs | *Super Puzzle Fighter II* (`AZ8E`/`AZ8P`) | `GBAMOD30` | Exact `V3EarlyVol64` PCM8 mixer **+** 2x-upsampled high-nibble-first 4-bit IMA ADPCM ARM mixer (`((nib & 7) * step) >> 2`) | **Yes** |
| `V3Standard` | 47 ROMs | *NFS: Porsche Unleashed / UG2 / MW / Carbon*, *Medal of Honor: Infiltrator*, *Driv3r*, *Asterix & Obelix XXL*, *Alien Hominid*, *Totally Spies!*, *Bionicle*, *Garfield*, *Duel Masters* | `GBAMOD30` | Streamlined `0xA0`-byte Q20.12 mixer (bit-identical across all 47 ROMs; removed `vol == 64` fast path) | No |

---

## Quickstart

### Zero-Allocation Mixing (`#![no_std]`)

```rust
use logik_state::{AccuracySynth, LogikPcmVoice, LogikVersion};

// Raw signed 8-bit PCM waveform
let pcm: &[u8] = &[0, 32, 64, 96, 127, 96, 64, 32, 0, 224, 192, 160, 128, 160, 192, 224];

// V3Standard Q20.12 looped voice, 1.0x pitch (0x1000), volume 64
let mut voice = LogikPcmVoice::new_v3(0, 0, 0x1000, pcm.len() as u32, 0, pcm.len() as u32, 64);
let mut out = [0i8; 128];

AccuracySynth::mix_pcm_voice(LogikVersion::V3Standard, pcm, &mut voice, &mut out);
```

---

## Golden Test Methodology

Every mixing path in `AccuracySynth` is verified strictly byte-for-byte (`assert_eq!`, zero tolerance) against real ARM7TDMI execution traces captured in headless `libmgba.so.0.10` using clean-room synthetic deterministic waveforms:

- **15 ARM7TDMI vectors (1,856 bytes total)**:
  - `V1Mod1` (2 vectors, 256 bytes): resampled loop + 2-voice wrapping sum.
  - `V2Mod2Halfword` (3 vectors, 320 bytes): general loop, `vol == 64` fast path, `len == 2` guard.
  - `V2Mod2Word` (2 vectors, 256 bytes): 32-bit general loop + `vol == 64` fast path.
  - `V3EarlyVol64` (3 vectors, 384 bytes): Q20.12 general loop, one-shot early stop, and `vol == 64` post-loop stale-`r6` quirk.
  - `V3EarlyVol64Adpcm` (3 vectors, 384 bytes): single 4-bit IMA ADPCM voice, PCM8 + ADPCM 2-voice mix, and odd-count (`63 + 65`) ADPCM resume.
  - `V3Standard` (2 vectors, 256 bytes): Q20.12 looped `vol == 64` (no quirk) + 2-voice looped/one-shot wrapping sum.

---

## Benchmarks

```sh
cargo bench --bench bench_audio
```

| Benchmark | Throughput |
| --- | ---: |
| `AccuracySynth::mix_v3_standard` (128f) | ~2.25 GSamples/s |
| `AccuracySynth::mix_v2_halfword` (`vol == 64`, 128f) | ~3.45 GSamples/s |
| `AccuracySynth::mix_v3_adpcm_voice` (2x IMA ADPCM, 128f) | ~1.13 GSamples/s |
| `StudioSynth::render_voice_f32` (48 kHz Lanczos-3, 1,024f) | ~396.8 MSamples/s |

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
