//! # logik-state-rs
//!
//! Pure Rust, zero-dependency parser, software synthesizer, and byte-exact golden-tested
//! emulator for the **Logik State GBAModPlay** sound engine on the Game Boy Advance.
//!
//! Covers **60 commercial GBA releases** across **6 distinct ARM7TDMI mixer revisions**:
//! - [`LogikVersion::V1Mod1`] (1 ROM: *Pinball Challenge Deluxe* `APLP`, `GBAMOD1.`)
//! - [`LogikVersion::V2Mod2Halfword`] (4 ROMs: *Aero the Acro-Bat*, `GBAMOD2.`)
//! - [`LogikVersion::V2Mod2Word`] (2 ROMs: *Zapper: One Wicked Cricket!*, `GBAMOD2.`)
//! - [`LogikVersion::V3EarlyVol64`] (3 ROMs: *Stuntman*, *2-in-1: V-Rally 3 + Stuntman*, `GBAMOD30`)
//! - [`LogikVersion::V3EarlyVol64Adpcm`] (3 ROMs: *Super Puzzle Fighter II*, `GBAMOD30` + 4-bit IMA ADPCM ARM mixer)
//! - [`LogikVersion::V3Standard`] (47 ROMs: *Need for Speed* series, *Medal of Honor: Infiltrator*,
//!   *Driv3r*, *Asterix & Obelix XXL*, *Alien Hominid*, *Totally Spies! 1 & 2*, *Bionicle*,
//!   *Lego Knights' Kingdom*, *Garfield*, *Duel Masters*, *Hot Wheels*, *Family Feud*, etc.)
//!
//! ## Features
//! - **Zero dependencies** in `[dependencies]`, `#![no_std]` compatible, `#![deny(unsafe_code)]`.
//! - **Dual synthesis pipelines**:
//!   - [`AccuracySynth`]: 100% bit-exact emulation of all 6 Logik State ARM7TDMI mixer revisions
//!     (PCM8 + 2x-upsampled 4-bit IMA ADPCM).
//!   - [`StudioSynth`]: 48,000 Hz `f32` band-limited Lanczos-3 polyphase windowed-sinc renderer
//!     with equal-power stereo panning.
//! - **Byte-exact `libmgba` ARM golden tested**: 1,856 / 1,856 output bytes verified strictly
//!   byte-for-byte (`assert_eq!`, zero tolerance) across all 6 engine revisions.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod error;
pub mod freq;
pub mod sample;
pub mod synth;
pub mod version;
pub mod voice;

pub use error::{Error, Result};
pub use freq::{
    freq_to_step_q12, rate_from_timer_reload, step_q12_to_freq, GBA_CLOCK_HZ, IMA_INDEX_TABLE,
    IMA_STEP_TABLE, LOGIK_DEFAULT_MIX_RATE_HZ, LOGIK_TIMER0_RELOADS_V2_ADPCM,
    LOGIK_TIMER0_RELOADS_V3, LOGIK_TIMER0_RELOADS_V3B, Q12_ONE,
};
#[cfg(feature = "alloc")]
pub use sample::LogikSample;
pub use sample::{
    decode_ima_adpcm4_arm_to_pcm8, decode_ima_adpcm4_standard_to_pcm8, LogikModHeader,
    LogikSampleDescV2, LogikSampleDescV3, LogikSampleView, GBAMOD1_MAGIC, GBAMOD2_MAGIC,
    GBAMOD30_MAGIC, LOGIK_DESC_V2_SIZE, LOGIK_DESC_V3_SIZE,
};
pub use synth::{AccuracySynth, StudioSynth};
pub use version::{LogikBankFormat, LogikSampleCodec, LogikVersion};
pub use voice::{LogikAdpcmVoice, LogikPcmVoice};
