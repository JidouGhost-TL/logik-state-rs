//! Hardware voice state structs for Logik State GBAModPlay PCM8 and 4-bit IMA ADPCM mixers.

/// Hardware PCM8 voice state matching Logik State's ARM7TDMI voice control block.
///
/// Supports both:
/// - **v1.x / v2.x integer-length voices** ([`crate::LogikVersion::V1Mod1`],
///   [`crate::LogikVersion::V2Mod2Halfword`], [`crate::LogikVersion::V2Mod2Word`]), where
///   `length` and `loop_len` are in whole samples.
/// - **v3.0+ Q20.12 fixed-point voices** ([`crate::LogikVersion::V3EarlyVol64`],
///   [`crate::LogikVersion::V3EarlyVol64Adpcm`], [`crate::LogikVersion::V3Standard`]), where
///   `length` and `loop_len` are in Q20.12 fixed-point (`samples << 12`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogikPcmVoice {
    /// Current base byte offset into the PCM8 sample buffer (`r3`).
    pub ptr_offset: usize,
    /// Current Q20.12 fixed-point playback phase (`r4`).
    pub phase_q12: u32,
    /// Q20.12 fixed-point phase increment per output sample (`r5`, `0x1000` = `1.0x`).
    pub step_q12: u32,
    /// Active segment length (`sl` — integer samples for v1/v2, Q20.12 for v3).
    pub length: u32,
    /// Byte offset added to `ptr_offset` on the first loop wrap (`ip`), then zeroed.
    pub loop_start: u32,
    /// Subsequent loop segment length (`r0` — integer samples for v1/v2, Q20.12 for v3;
    /// `0` denotes a one-shot voice in v3).
    pub loop_len: u32,
    /// Channel volume (`sb`, typically `0..=64`; `64` triggers the `asr #2` fast path on
    /// revisions that include it).
    pub volume: u32,
    /// Whether this voice is currently active.
    pub active: bool,
}

impl LogikPcmVoice {
    /// Create a new v1.x / v2.x integer-length PCM8 voice (`V1Mod1`, `V2Mod2Halfword`,
    /// `V2Mod2Word`).
    #[inline]
    #[must_use]
    pub const fn new_v1_v2(
        ptr_offset: usize,
        phase_q12: u32,
        step_q12: u32,
        length_samples: u32,
        loop_start_bytes: u32,
        loop_len_samples: u32,
        volume: u32,
    ) -> Self {
        Self {
            ptr_offset,
            phase_q12,
            step_q12,
            length: length_samples,
            loop_start: loop_start_bytes,
            loop_len: loop_len_samples,
            volume,
            active: true,
        }
    }

    /// Create a new v3.0+ Q20.12 fixed-point PCM8 voice (`V3EarlyVol64`, `V3EarlyVol64Adpcm`,
    /// `V3Standard`).
    ///
    /// `length_samples` and `loop_len_samples` are automatically converted to Q20.12 (`<< 12`).
    /// Pass `loop_len_samples = 0` for a one-shot sample.
    #[inline]
    #[must_use]
    pub const fn new_v3(
        ptr_offset: usize,
        phase_q12: u32,
        step_q12: u32,
        length_samples: u32,
        loop_start_bytes: u32,
        loop_len_samples: u32,
        volume: u32,
    ) -> Self {
        Self {
            ptr_offset,
            phase_q12,
            step_q12,
            length: length_samples << 12,
            loop_start: loop_start_bytes,
            loop_len: loop_len_samples << 12,
            volume,
            active: length_samples != 0,
        }
    }
}

/// Hardware 4-bit IMA ADPCM voice state matching the `V3EarlyVol64Adpcm` ARM7TDMI mixer
/// (*Super Puzzle Fighter II*, `AZ8E` / `AZ8P`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogikAdpcmVoice {
    /// Remaining ADPCM nibbles to decode (`[r0, #0x0C]`; each nibble produces 2 output samples).
    pub remaining_nibbles: u32,
    /// Channel volume (`[r0, #0x18]`; scaled via `((pred * vol) as u32) >> 16`).
    pub volume: u32,
    /// Current unclamped 32-bit IMA ADPCM predictor (`[r0, #0x1C]`).
    pub predictor: i32,
    /// Current IMA step-table index (`[r0, #0x20]`, clamped to `0..=88`).
    pub step_index: i32,
    /// Current byte offset into the encoded ADPCM stream (`[r0, #0x24]`).
    pub byte_offset: usize,
    /// Nibble parity toggle (`[r0, #0x28]`: `0` = fetch next byte & decode high nibble;
    /// non-zero = decode low nibble of `cached_byte`).
    pub nibble_toggle: u32,
    /// Cached current ADPCM byte (`[r0, #0x2C]`).
    pub cached_byte: u8,
    /// Whether this voice is currently active.
    pub active: bool,
}

impl LogikAdpcmVoice {
    /// Create a new 4-bit IMA ADPCM voice positioned at `byte_offset` with `remaining_nibbles`
    /// nibbles to decode (producing `2 * remaining_nibbles` output samples).
    #[inline]
    #[must_use]
    pub const fn new(byte_offset: usize, remaining_nibbles: u32, volume: u32) -> Self {
        Self {
            remaining_nibbles,
            volume,
            predictor: 0,
            step_index: 0,
            byte_offset,
            nibble_toggle: 0,
            cached_byte: 0,
            active: remaining_nibbles != 0,
        }
    }
}
