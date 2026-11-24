//! Zero-allocation parsers for Logik State GBAModPlay song headers, 20-byte V2 and
//! 24-byte V3 sample descriptors, and 4-bit IMA ADPCM decoders.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::freq::{IMA_INDEX_TABLE, IMA_STEP_TABLE, LOGIK_DEFAULT_MIX_RATE_HZ};
use crate::version::{LogikBankFormat, LogikSampleCodec};

/// Size of a v2.x / early v3.0 [`LogikSampleDescV2`] descriptor in bytes (`0x14`).
pub const LOGIK_DESC_V2_SIZE: usize = 20;

/// Size of a v3.0b / v3.1 [`LogikSampleDescV3`] descriptor in bytes (`0x18`).
pub const LOGIK_DESC_V3_SIZE: usize = 24;

/// 8-byte song module magic for GBAModPlay v1.x (*Pinball Challenge Deluxe*).
pub const GBAMOD1_MAGIC: &[u8; 8] = b"GBAMOD1.";

/// 8-byte song module magic for GBAModPlay v2.x (*Aero the Acro-Bat*, *Zapper*).
pub const GBAMOD2_MAGIC: &[u8; 8] = b"GBAMOD2.";

/// 8-byte song module magic for GBAModPlay v3.0+ (53 corpus titles).
pub const GBAMOD30_MAGIC: &[u8; 8] = b"GBAMOD30";

/// Parsed `GBAMOD1.` / `GBAMOD2.` / `GBAMOD30` song module header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogikModHeader {
    /// Module format magic (`"GBAMOD1."`, `"GBAMOD2."`, or `"GBAMOD30"`).
    pub magic: [u8; 8],
    /// Number of tracker channels declared in the header.
    pub num_channels: u32,
    /// Native hardware mixing rate in Hz (e.g. `10_512` or `21_024`).
    pub mix_rate_hz: u32,
    /// Number of packed patterns declared in the header.
    pub num_patterns: u32,
}

impl LogikModHeader {
    /// Parse a `GBAMOD1.`, `GBAMOD2.`, or `GBAMOD30` module header from the start of `data`.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::UnexpectedEof);
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&data[0..8]);

        if &magic == GBAMOD30_MAGIC {
            if data.len() < 0x3C {
                return Err(Error::UnexpectedEof);
            }
            let num_channels = read_u32_le(data, 0x28);
            let mix_rate_hz = read_u32_le(data, 0x34);
            let num_patterns = read_u32_le(data, 0x38);
            return Ok(Self {
                magic,
                num_channels,
                mix_rate_hz,
                num_patterns,
            });
        }

        if &magic == GBAMOD2_MAGIC || &magic == GBAMOD1_MAGIC {
            if data.len() < 0x1C {
                return Err(Error::UnexpectedEof);
            }
            let num_channels = read_u32_le(data, 0x08);
            let mix_rate_hz = read_u32_le(data, 0x14);
            let num_patterns = read_u32_le(data, 0x18);
            return Ok(Self {
                magic,
                num_channels,
                mix_rate_hz,
                num_patterns,
            });
        }

        Err(Error::InvalidHeader(
            "expected GBAMOD1., GBAMOD2., or GBAMOD30 magic",
        ))
    }
}

/// 20-byte Logik State GBAModPlay v2.x / early v3.0 sample descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogikSampleDescV2 {
    /// Byte offset relative to the sample bank's packed payload base.
    pub data_offset: u32,
    /// Encoded payload length in bytes (PCM8 bytes or 4-bit IMA ADPCM bytes).
    pub length: u32,
    /// Packed volume/flags word (default volume in bits `16..=23`, typically `0x0040_0000` = 64).
    pub vol_flags: u32,
    /// Loop start offset in bytes.
    pub loop_start: u32,
    /// Loop length in bytes (`0` = one-shot, `1` with `loop_start == 0` = IMA ADPCM, `> 2` = looped).
    pub loop_len: u32,
}

impl LogikSampleDescV2 {
    /// Parse a 20-byte `LogikSampleDescV2` from `data[0..20]`.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < LOGIK_DESC_V2_SIZE {
            return Err(Error::UnexpectedEof);
        }
        Ok(Self {
            data_offset: read_u32_le(data, 0),
            length: read_u32_le(data, 4),
            vol_flags: read_u32_le(data, 8),
            loop_start: read_u32_le(data, 12),
            loop_len: read_u32_le(data, 16),
        })
    }

    /// Default sample volume (`0..=64`, extracted from bits `16..=23` of `vol_flags`).
    #[inline]
    #[must_use]
    pub const fn default_volume(&self) -> u8 {
        ((self.vol_flags >> 16) & 0xFF) as u8
    }

    /// Whether this descriptor uses the Logik State 4-bit IMA ADPCM marker
    /// (`loop_start == 0 && loop_len == 1`).
    #[inline]
    #[must_use]
    pub const fn is_ima_adpcm(&self) -> bool {
        self.loop_start == 0 && self.loop_len == 1
    }

    /// Whether this descriptor represents a looped PCM8 waveform.
    #[inline]
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        !self.is_ima_adpcm()
            && self.loop_len > 2
            && self.loop_start.saturating_add(self.loop_len) <= self.length
    }
}

/// 24-byte Logik State GBAModPlay v3.0b / v3.1 sample descriptor with explicit sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogikSampleDescV3 {
    /// Byte offset relative to the sample bank's packed payload base.
    pub data_offset: u32,
    /// Encoded payload length in bytes.
    pub length: u32,
    /// Packed volume/flags word (default volume in bits `16..=23`, typically `0x0040_0000` = 64).
    pub vol_flags: u32,
    /// Loop start offset in bytes.
    pub loop_start: u32,
    /// Explicit native playback sample rate in Hz (e.g. `10_512`).
    pub sample_rate_hz: u32,
    /// Loop length in bytes (`0` = one-shot, `> 2` = looped).
    pub loop_len: u32,
}

impl LogikSampleDescV3 {
    /// Parse a 24-byte `LogikSampleDescV3` from `data[0..24]`.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < LOGIK_DESC_V3_SIZE {
            return Err(Error::UnexpectedEof);
        }
        Ok(Self {
            data_offset: read_u32_le(data, 0),
            length: read_u32_le(data, 4),
            vol_flags: read_u32_le(data, 8),
            loop_start: read_u32_le(data, 12),
            sample_rate_hz: read_u32_le(data, 16),
            loop_len: read_u32_le(data, 20),
        })
    }

    /// Default sample volume (`0..=64`, extracted from bits `16..=23` of `vol_flags`).
    #[inline]
    #[must_use]
    pub const fn default_volume(&self) -> u8 {
        ((self.vol_flags >> 16) & 0xFF) as u8
    }

    /// Whether this descriptor represents a looped PCM8 waveform.
    #[inline]
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        self.loop_len > 2 && self.loop_start.saturating_add(self.loop_len) <= self.length
    }
}

/// Zero-allocation borrowed view of a Logik State GBAModPlay PCM8 sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogikSampleView<'a> {
    /// Sample bank descriptor format (`V2Desc20` or `V3Desc24`).
    pub bank_format: LogikBankFormat,
    /// Source codec (`Pcm8` or `ImaAdpcm4`).
    pub codec: LogikSampleCodec,
    /// Native playback sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Default volume (`0..=64`).
    pub default_volume: u8,
    /// Loop start in samples (`None` for one-shot).
    pub loop_start: Option<u32>,
    /// Loop end in samples (`None` for one-shot).
    pub loop_end: Option<u32>,
    /// Raw signed 8-bit PCM bytes (stored as `u8` bit-patterns of `i8`).
    pub pcm_bytes: &'a [u8],
}

impl<'a> LogikSampleView<'a> {
    /// Construct a `LogikSampleView` directly from raw signed 8-bit PCM bytes.
    #[inline]
    #[must_use]
    pub const fn from_pcm_bytes(
        sample_rate_hz: u32,
        default_volume: u8,
        loop_start: Option<u32>,
        loop_end: Option<u32>,
        pcm_bytes: &'a [u8],
    ) -> Self {
        Self {
            bank_format: LogikBankFormat::V3Desc24,
            codec: LogikSampleCodec::Pcm8,
            sample_rate_hz,
            default_volume,
            loop_start,
            loop_end,
            pcm_bytes,
        }
    }

    /// Construct a `LogikSampleView` from a [`LogikSampleDescV3`] and the sample bank's
    /// packed payload slice (`data_base..`).
    pub fn from_v3_desc(desc: &LogikSampleDescV3, payload: &'a [u8]) -> Result<Self> {
        let start = desc.data_offset as usize;
        let len = desc.length as usize;
        let end = start.checked_add(len).ok_or(Error::UnexpectedEof)?;
        let pcm_bytes = payload.get(start..end).ok_or(Error::UnexpectedEof)?;
        let rate = if (4_000..=48_000).contains(&desc.sample_rate_hz) {
            desc.sample_rate_hz
        } else {
            LOGIK_DEFAULT_MIX_RATE_HZ
        };
        let (loop_start, loop_end) = if desc.is_looped() {
            (Some(desc.loop_start), Some(desc.loop_start + desc.loop_len))
        } else {
            (None, None)
        };
        Ok(Self {
            bank_format: LogikBankFormat::V3Desc24,
            codec: LogikSampleCodec::Pcm8,
            sample_rate_hz: rate,
            default_volume: desc.default_volume(),
            loop_start,
            loop_end,
            pcm_bytes,
        })
    }

    /// Construct a `LogikSampleView` from a [`LogikSampleDescV2`] and the sample bank's
    /// packed payload slice (`data_base..`).
    pub fn from_v2_desc(
        desc: &LogikSampleDescV2,
        payload: &'a [u8],
        default_rate_hz: u32,
    ) -> Result<Self> {
        let start = desc.data_offset as usize;
        let len = desc.length as usize;
        let end = start.checked_add(len).ok_or(Error::UnexpectedEof)?;
        let pcm_bytes = payload.get(start..end).ok_or(Error::UnexpectedEof)?;
        let (loop_start, loop_end) = if desc.is_looped() {
            (Some(desc.loop_start), Some(desc.loop_start + desc.loop_len))
        } else {
            (None, None)
        };
        Ok(Self {
            bank_format: LogikBankFormat::V2Desc20,
            codec: if desc.is_ima_adpcm() {
                LogikSampleCodec::ImaAdpcm4
            } else {
                LogikSampleCodec::Pcm8
            },
            sample_rate_hz: default_rate_hz,
            default_volume: desc.default_volume(),
            loop_start,
            loop_end,
            pcm_bytes,
        })
    }

    /// Number of bytes in `pcm_bytes`.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.pcm_bytes.len()
    }

    /// Whether `pcm_bytes` is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.pcm_bytes.is_empty()
    }

    /// Whether this sample has a valid loop range.
    #[inline]
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        self.loop_start.is_some() && self.loop_end.is_some()
    }
}

/// Owned Logik State GBAModPlay decoded PCM8 sample (requires `alloc`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogikSample {
    /// Sample bank descriptor format (`V2Desc20` or `V3Desc24`).
    pub bank_format: LogikBankFormat,
    /// Original source codec (`Pcm8` or `ImaAdpcm4`).
    pub codec: LogikSampleCodec,
    /// Native playback sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Default volume (`0..=64`).
    pub default_volume: u8,
    /// Loop start in samples (`None` for one-shot).
    pub loop_start: Option<u32>,
    /// Loop end in samples (`None` for one-shot).
    pub loop_end: Option<u32>,
    /// Decoded signed 8-bit PCM bytes (stored as `u8` bit-patterns of `i8`).
    pub pcm_bytes: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl LogikSample {
    /// Borrow this owned sample as a [`LogikSampleView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> LogikSampleView<'_> {
        LogikSampleView {
            bank_format: self.bank_format,
            codec: self.codec,
            sample_rate_hz: self.sample_rate_hz,
            default_volume: self.default_volume,
            loop_start: self.loop_start,
            loop_end: self.loop_end,
            pcm_bytes: &self.pcm_bytes,
        }
    }
}

/// Decode 4-bit IMA ADPCM using the exact **Logik State `V3EarlyVol64Adpcm` ARM7TDMI**
/// algorithm (*Super Puzzle Fighter II*, `AZ8E`/`AZ8P`):
///
/// - **High nibble first**, then low nibble.
/// - `diff = (((nib & 7) * step) as u32) >> 2` (no `+ step >> 3` bias).
/// - Unclamped 32-bit `predictor`, shifted via `((pred * vol) as u32) >> 16`.
/// - Writes 2 output samples per byte into `out` (1 sample per nibble, without 2x doubling;
///   use [`crate::AccuracySynth::mix_v3_adpcm_voice`] for the exact 2x-doubled hardware mix).
///
/// Returns the number of decoded samples written to `out`.
pub fn decode_ima_adpcm4_arm_to_pcm8(raw: &[u8], volume: u32, out: &mut [i8]) -> usize {
    let mut pred: i32 = 0;
    let mut step_idx: i32 = 0;
    let mut step: i32 = IMA_STEP_TABLE[0];
    let vol = volume as i32;
    let mut written = 0usize;

    for &byte in raw {
        for nib in [(byte >> 4) & 0x0F, byte & 0x0F] {
            if written >= out.len() {
                return written;
            }
            step_idx = (step_idx + IMA_INDEX_TABLE[nib as usize]).clamp(0, 88);
            let diff = ((((nib & 7) as i32).wrapping_mul(step)) as u32 >> 2) as i32;
            if (nib & 8) != 0 {
                pred = pred.wrapping_sub(diff);
            } else {
                pred = pred.wrapping_add(diff);
            }
            step = IMA_STEP_TABLE[step_idx as usize];
            out[written] = ((pred.wrapping_mul(vol) as u32) >> 16) as i8;
            written += 1;
        }
    }

    written
}

/// Decode standard low-nibble-first 4-bit IMA ADPCM (`pred` clamped to `[-32768, 32767]`,
/// shifted right by 8) into `out`.
///
/// Returns the number of decoded samples written to `out`.
pub fn decode_ima_adpcm4_standard_to_pcm8(raw: &[u8], out: &mut [i8]) -> usize {
    let mut pred: i32 = 0;
    let mut step_idx: i32 = 0;
    let mut written = 0usize;

    for &byte in raw {
        for nib in [byte & 0x0F, (byte >> 4) & 0x0F] {
            if written >= out.len() {
                return written;
            }
            let step = IMA_STEP_TABLE[step_idx as usize];
            let mut diff = step >> 3;
            if (nib & 1) != 0 {
                diff += step >> 2;
            }
            if (nib & 2) != 0 {
                diff += step >> 1;
            }
            if (nib & 4) != 0 {
                diff += step;
            }
            if (nib & 8) != 0 {
                pred -= diff;
            } else {
                pred += diff;
            }
            pred = pred.clamp(-32_768, 32_767);
            step_idx = (step_idx + IMA_INDEX_TABLE[nib as usize]).clamp(0, 88);
            out[written] = (pred >> 8) as i8;
            written += 1;
        }
    }

    written
}

#[inline]
fn read_u32_le(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_mod_headers() {
        let mut v3 = [0u8; 0x40];
        v3[0..8].copy_from_slice(GBAMOD30_MAGIC);
        v3[0x28..0x2C].copy_from_slice(&8u32.to_le_bytes());
        v3[0x34..0x38].copy_from_slice(&13_379u32.to_le_bytes());
        v3[0x38..0x3C].copy_from_slice(&24u32.to_le_bytes());
        let h3 = LogikModHeader::parse(&v3).expect("valid GBAMOD30");
        assert_eq!(h3.num_channels, 8);
        assert_eq!(h3.mix_rate_hz, 13_379);
        assert_eq!(h3.num_patterns, 24);

        let mut v2 = [0u8; 0x20];
        v2[0..8].copy_from_slice(GBAMOD2_MAGIC);
        v2[0x08..0x0C].copy_from_slice(&4u32.to_le_bytes());
        v2[0x14..0x18].copy_from_slice(&21_024u32.to_le_bytes());
        v2[0x18..0x1C].copy_from_slice(&12u32.to_le_bytes());
        let h2 = LogikModHeader::parse(&v2).expect("valid GBAMOD2.");
        assert_eq!(h2.num_channels, 4);
        assert_eq!(h2.mix_rate_hz, 21_024);

        assert!(LogikModHeader::parse(b"INVALID!").is_err());
    }

    #[test]
    fn test_parse_v2_and_v3_descriptors() {
        let mut d2 = [0u8; LOGIK_DESC_V2_SIZE];
        d2[0..4].copy_from_slice(&0u32.to_le_bytes());
        d2[4..8].copy_from_slice(&64u32.to_le_bytes());
        d2[8..12].copy_from_slice(&0x0040_0000u32.to_le_bytes());
        d2[12..16].copy_from_slice(&0u32.to_le_bytes());
        d2[16..20].copy_from_slice(&1u32.to_le_bytes());
        let parsed2 = LogikSampleDescV2::parse(&d2).expect("valid v2");
        assert_eq!(parsed2.default_volume(), 64);
        assert!(parsed2.is_ima_adpcm());
        assert!(!parsed2.is_looped());

        let mut d3 = [0u8; LOGIK_DESC_V3_SIZE];
        d3[0..4].copy_from_slice(&0u32.to_le_bytes());
        d3[4..8].copy_from_slice(&96u32.to_le_bytes());
        d3[8..12].copy_from_slice(&0x0038_0000u32.to_le_bytes());
        d3[12..16].copy_from_slice(&16u32.to_le_bytes());
        d3[16..20].copy_from_slice(&15_768u32.to_le_bytes());
        d3[20..24].copy_from_slice(&64u32.to_le_bytes());
        let parsed3 = LogikSampleDescV3::parse(&d3).expect("valid v3");
        assert_eq!(parsed3.default_volume(), 0x38);
        assert!(parsed3.is_looped());
        let payload = [0u8; 96];
        let view = LogikSampleView::from_v3_desc(&parsed3, &payload).expect("valid view");
        assert_eq!(view.sample_rate_hz, 15_768);
        assert_eq!(view.loop_start, Some(16));
        assert_eq!(view.loop_end, Some(80));
    }

    #[test]
    fn test_ima_standard_decoder() {
        let mut out = [0i8; 6];
        let n = decode_ima_adpcm4_standard_to_pcm8(&[0x77, 0x77, 0xF7], &mut out);
        assert_eq!(n, 6);
        assert_eq!(out, [0, 0, 0, 0, 2, -1]);
    }
}
