//! Logik State GBAModPlay engine revision detection and mixer variant metadata.
//!
//! A full-corpus audit of all 3,374 GBA ROMs identifies **60 commercial releases** built on
//! Logik State's GBAModPlay driver spanning **6 distinct ARM7TDMI mixer revisions**:
//!
//! 1. [`LogikVersion::V1Mod1`] (**1 ROM** — *Pinball Challenge Deluxe* `APLP`, `GBAMOD1.`):
//!    - 16-bit halfword length/loop fields (`ldrh`), unsigned `lsr r6, r4, #12` index extraction,
//!      no `len == 2` early-exit guard, `and r4, r4, #0x0FFF` loop wrap, and `lsr #8` / `lsr #2`
//!      volume shifts (scalar + 8x-unrolled routines).
//! 2. [`LogikVersion::V2Mod2Halfword`] (**4 ROMs** — *Aero the Acro-Bat* `AAOE`/`AAOJ`/`AAOP`,
//!    `GBAMOD2.`):
//!    - 16-bit halfword length/loop fields (`ldrh`), `cmp sl, #2` early-exit guard, `vol == 64`
//!      (`asr #2`) fast path, signed `asr r6, r4, #12` index extraction, and `and r4, #0x0FFF`
//!      loop wrap.
//! 3. [`LogikVersion::V2Mod2Word`] (**2 ROMs** — *Zapper: One Wicked Cricket!* `AZPE`/`AZPP`,
//!    `GBAMOD2.`):
//!    - Upgraded 32-bit word length/loop fields (`ldr`) with volume at `+0x18`, retaining the
//!      v2 `cmp sl, #2` early-exit guard, integer-index `cmp r6, sl`, `and r4, #0x0FFF` loop wrap,
//!      and `vol == 64` fast path.
//! 4. [`LogikVersion::V3EarlyVol64`] (**3 ROMs** — *Stuntman* `AUXE`/`AUXP`, *2-in-1: V-Rally 3 +
//!    Stuntman* `BCSP`, `GBAMOD30`):
//!    - 32-bit Q20.12 fixed-point length/loop comparison (`cmp r4, sl`), `tst sl, sl` early-exit
//!      guard, phase-preserving `sub r4, r4, sl` loop wrap, and `vol == 64` fast path (which
//!      exhibits a real hardware quirk where the first sample after a loop wrap reuses the
//!      pre-wrap `r6` index).
//! 5. [`LogikVersion::V3EarlyVol64Adpcm`] (**3 ROMs** — *Super Puzzle Fighter II* `AZ8E`/`AZ8P`,
//!    `GBAMOD30`):
//!    - Exact [`LogikVersion::V3EarlyVol64`] PCM8 mixer **plus** a dedicated ARM7TDMI 4-bit IMA
//!      ADPCM mixer (high-nibble-first, `((nib & 7) * step) >> 2`, unclamped `i32` predictor,
//!      `(pred * vol) >> 16`, and 2x sample-hold upsampling).
//! 6. [`LogikVersion::V3Standard`] (**47 ROMs** — *Need for Speed: Porsche Unleashed*, *NFS:
//!    Underground 2*, *NFS: Most Wanted*, *NFS: Carbon*, *Medal of Honor: Infiltrator*, *Driv3r*,
//!    *Asterix & Obelix XXL*, *Alien Hominid*, *Totally Spies! 1 & 2*, *Bionicle*, *Lego Knights'
//!    Kingdom*, *Garfield*, *Duel Masters*, *Hot Wheels*, *Family Feud*, *Kid Paddle*, *Premier
//!    Manager*, etc., `GBAMOD30`):
//!    - Streamlined `0xA0`-byte ARM7TDMI mixer: bit-for-bit identical across all 47 ROMs. Removes
//!      the buggy `vol == 64` fast path in favor of a single unified Q20.12 loop with `tst sl, sl`,
//!      `cmp r4, sl`, `sub r4, r4, sl`, and `mul r7, sb, r7; add r7, r8, r7, asr #8`.

/// Sample-bank descriptor layout used by a Logik State GBAModPlay ROM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogikBankFormat {
    /// 20-byte `LogikSampleDescV2` descriptors (v2.x / early v3.0; mix rate from module header).
    V2Desc20,
    /// 24-byte `LogikSampleDescV3` descriptors (v3.0b / v3.1; explicit per-sample rate in Hz).
    V3Desc24,
}

/// Sample encoding format supported by Logik State GBAModPlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogikSampleCodec {
    /// Raw signed 8-bit linear PCM (`i8`).
    Pcm8,
    /// 4-bit IMA ADPCM (decoded by the dedicated ARM7TDMI ADPCM mixer in `V3EarlyVol64Adpcm`).
    ImaAdpcm4,
}

/// Distinct Logik State GBAModPlay ARM7TDMI mixer revisions discovered across the GBA ROM corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum LogikVersion {
    /// GBAModPlay v1.x (`GBAMOD1.`, 1 ROM: *Pinball Challenge Deluxe* `APLP`):
    /// 16-bit halfword length/loop, unsigned `lsr #12`, `and #0xFFF` wrap, no `len == 2` guard.
    V1Mod1,
    /// GBAModPlay v2.x halfword (`GBAMOD2.`, 4 ROMs: *Aero the Acro-Bat* `AAOE`/`AAOJ`/`AAOP`):
    /// 16-bit halfword length/loop, `cmp sl, #2` guard, `vol == 64` fast path, `and #0xFFF` wrap.
    V2Mod2Halfword,
    /// GBAModPlay v2.x word (`GBAMOD2.`, 2 ROMs: *Zapper* `AZPE`/`AZPP`):
    /// 32-bit word length/loop, `cmp sl, #2` guard, `vol == 64` fast path, `and #0xFFF` wrap.
    V2Mod2Word,
    /// GBAModPlay early v3.0 (`GBAMOD30`, 3 ROMs: *Stuntman* `AUXE`/`AUXP`, *2-in-1* `BCSP`):
    /// Q20.12 length/loop, `tst sl, sl` guard, `sub r4, r4, sl` wrap, `vol == 64` fast path
    /// (with post-loop stale-`r6` quirk).
    V3EarlyVol64,
    /// GBAModPlay early v3.0 + IMA ADPCM (`GBAMOD30`, 3 ROMs: *Super Puzzle Fighter II* `AZ8E`/`AZ8P`):
    /// Exact [`Self::V3EarlyVol64`] PCM mixer plus the 2x-upsampled high-nibble-first 4-bit IMA
    /// ADPCM ARM7TDMI mixer.
    V3EarlyVol64Adpcm,
    /// GBAModPlay standard v3.0b/v3.1 (`GBAMOD30`, 47 ROMs: *NFS* series, *Medal of Honor:
    /// Infiltrator*, *Driv3r*, *Asterix & Obelix XXL*, *Alien Hominid*, *Totally Spies!*, etc.):
    /// Streamlined `0xA0`-byte Q20.12 ARM7TDMI mixer without the `vol == 64` fast path.
    #[default]
    V3Standard,
}

impl LogikVersion {
    /// Whether this revision compares phase directly in Q20.12 fixed point (`cmp r4, sl`)
    /// and preserves loop overshoot via `sub r4, r4, sl`.
    #[inline]
    #[must_use]
    pub const fn uses_q12_length(self) -> bool {
        matches!(
            self,
            Self::V3EarlyVol64 | Self::V3EarlyVol64Adpcm | Self::V3Standard
        )
    }

    /// Whether this revision uses 16-bit halfword length/loop fields (`ldrh`) in its voice struct.
    #[inline]
    #[must_use]
    pub const fn uses_halfword_lengths(self) -> bool {
        matches!(self, Self::V1Mod1 | Self::V2Mod2Halfword)
    }

    /// Whether this revision includes the dedicated `vol == 64` (`0x40`) `asr #2` fast path.
    #[inline]
    #[must_use]
    pub const fn has_vol64_fast_path(self) -> bool {
        matches!(
            self,
            Self::V2Mod2Halfword
                | Self::V2Mod2Word
                | Self::V3EarlyVol64
                | Self::V3EarlyVol64Adpcm
        )
    }

    /// Whether this revision exhibits the `V3EarlyVol64` post-loop stale-`r6` quirk when
    /// `volume == 64`.
    #[inline]
    #[must_use]
    pub const fn has_vol64_loop_quirk(self) -> bool {
        matches!(self, Self::V3EarlyVol64 | Self::V3EarlyVol64Adpcm)
    }

    /// Whether this revision includes the dedicated ARM7TDMI 4-bit IMA ADPCM mixer.
    #[inline]
    #[must_use]
    pub const fn has_ima_adpcm_mixer(self) -> bool {
        matches!(self, Self::V3EarlyVol64Adpcm)
    }

    /// Byte size of the primary ARM7TDMI PCM mixer routine in ROM/IWRAM.
    #[inline]
    #[must_use]
    pub const fn pcm_mixer_size_bytes(self) -> usize {
        match self {
            Self::V1Mod1 => 0x88,
            Self::V2Mod2Halfword | Self::V2Mod2Word => 0x118,
            Self::V3EarlyVol64 | Self::V3EarlyVol64Adpcm => 0x120,
            Self::V3Standard => 0xA0,
        }
    }

    /// Detect the exact Logik State GBAModPlay ARM7TDMI mixer revision from a GBA ROM or
    /// IWRAM binary image.
    ///
    /// Verified 60 / 60 (100%) across all Logik State GBAModPlay ROMs in the GBA corpus.
    #[must_use]
    pub fn detect_from_bytes(data: &[u8]) -> Option<Self> {
        // Shared ARM7TDMI mixer prologue:
        //   push  {r1..r12}  (0xE92D1FFE)
        //   stmdb sp!, {r0}  (0xE92D0001)
        const PROLOGUE: [u8; 8] = [0xFE, 0x1F, 0x2D, 0xE9, 0x01, 0x00, 0x2D, 0xE9];
        // Shared inner-loop sample & accumulator fetch:
        //   ldrsb r7, [r3, r6] (0xE19370D6)
        //   ldrsb r8, [r1]     (0xE1D180D0)
        const INNER_SIG: [u8; 8] = [0xD6, 0x70, 0x93, 0xE1, 0xD0, 0x80, 0xD1, 0xE1];
        // Dedicated AZ8E/AZ8P 4-bit IMA ADPCM nibble-fetch pair:
        //   ldrb  sl, [r6], #1 (0xE4D6A001)
        //   lsr   r3, sl, #4   (0xE1A0322A)
        const IMA_ADPCM_SIG: [u8; 8] = [0x01, 0xA0, 0xD6, 0xE4, 0x2A, 0x32, 0xA0, 0xE1];

        let mut search_from = 0usize;
        while let Some(rel) = find_subslice(&data[search_from..], &PROLOGUE) {
            let pos = search_from + rel;
            search_from = pos + 4;
            if pos + 0x20 > data.len() {
                break;
            }
            let win_end = (pos + 0x100).min(data.len());
            if find_subslice(&data[pos..win_end], &INNER_SIG).is_none() {
                continue;
            }

            let w2 = read_u32_le(data, pos + 8);
            let w3 = read_u32_le(data, pos + 12);
            let w6 = read_u32_le(data, pos + 24);

            // 1. V1Mod1: `ldr r3, [r0]` (0xE5903000) immediately after prologue
            if w2 == 0xE590_3000 {
                return Some(Self::V1Mod1);
            }
            // 2. V2Mod2Halfword: `ldrh sl, [r0, #0xc]` (0xE1D0A0BC)
            if w2 == 0xE1D0_A0BC {
                return Some(Self::V2Mod2Halfword);
            }
            // 3. Word-sized length at `[r0, #0xc]`: `ldr sl, [r0, #0xc]` (0xE590A00C)
            if w2 == 0xE590_A00C {
                // V2Mod2Word: `cmp sl, #2` (0xE35A0002)
                if w3 == 0xE35A_0002 {
                    return Some(Self::V2Mod2Word);
                }
                // V3Standard: `tst sl, sl` + no `cmp sb, #0x40` (`ldr r3, [r0]` at +0x18)
                if w6 == 0xE590_3000 {
                    return Some(Self::V3Standard);
                }
                // V3EarlyVol64 / V3EarlyVol64Adpcm: `cmp sb, #0x40` (0xE3590040) at +0x18
                if w6 == 0xE359_0040 {
                    if find_subslice(data, &IMA_ADPCM_SIG).is_some() {
                        return Some(Self::V3EarlyVol64Adpcm);
                    }
                    return Some(Self::V3EarlyVol64);
                }
            }
        }

        None
    }
}

#[inline]
fn read_u32_le(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

#[inline]
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stub(w2: u32, w3: u32, w6: u32, with_adpcm: bool) -> [u8; 96] {
        let mut buf = [0u8; 96];
        buf[0..4].copy_from_slice(&0xE92D_1FFEu32.to_le_bytes());
        buf[4..8].copy_from_slice(&0xE92D_0001u32.to_le_bytes());
        buf[8..12].copy_from_slice(&w2.to_le_bytes());
        buf[12..16].copy_from_slice(&w3.to_le_bytes());
        buf[24..28].copy_from_slice(&w6.to_le_bytes());
        buf[48..52].copy_from_slice(&0xE193_70D6u32.to_le_bytes());
        buf[52..56].copy_from_slice(&0xE1D1_80D0u32.to_le_bytes());
        if with_adpcm {
            buf[72..76].copy_from_slice(&0xE4D6_A001u32.to_le_bytes());
            buf[76..80].copy_from_slice(&0xE1A0_322Au32.to_le_bytes());
        }
        buf
    }

    #[test]
    fn test_detect_all_six_revisions() {
        let v1 = make_stub(0xE590_3000, 0, 0, false);
        assert_eq!(LogikVersion::detect_from_bytes(&v1), Some(LogikVersion::V1Mod1));

        let v2_hw = make_stub(0xE1D0_A0BC, 0xE35A_0002, 0xE359_0040, false);
        assert_eq!(
            LogikVersion::detect_from_bytes(&v2_hw),
            Some(LogikVersion::V2Mod2Halfword)
        );

        let v2_w = make_stub(0xE590_A00C, 0xE35A_0002, 0xE359_0040, false);
        assert_eq!(
            LogikVersion::detect_from_bytes(&v2_w),
            Some(LogikVersion::V2Mod2Word)
        );

        let v3_early = make_stub(0xE590_A00C, 0xE11A_000A, 0xE359_0040, false);
        assert_eq!(
            LogikVersion::detect_from_bytes(&v3_early),
            Some(LogikVersion::V3EarlyVol64)
        );

        let v3_adpcm = make_stub(0xE590_A00C, 0xE11A_000A, 0xE359_0040, true);
        assert_eq!(
            LogikVersion::detect_from_bytes(&v3_adpcm),
            Some(LogikVersion::V3EarlyVol64Adpcm)
        );

        let v3_std = make_stub(0xE590_A00C, 0xE11A_000A, 0xE590_3000, false);
        assert_eq!(
            LogikVersion::detect_from_bytes(&v3_std),
            Some(LogikVersion::V3Standard)
        );

        assert_eq!(LogikVersion::detect_from_bytes(&[0u8; 64]), None);
    }

    #[test]
    fn test_version_metadata() {
        assert!(LogikVersion::V1Mod1.uses_halfword_lengths());
        assert!(!LogikVersion::V1Mod1.uses_q12_length());
        assert_eq!(LogikVersion::V1Mod1.pcm_mixer_size_bytes(), 0x88);

        assert!(LogikVersion::V2Mod2Halfword.has_vol64_fast_path());
        assert!(!LogikVersion::V2Mod2Word.uses_halfword_lengths());
        assert_eq!(LogikVersion::V2Mod2Word.pcm_mixer_size_bytes(), 0x118);

        assert!(LogikVersion::V3EarlyVol64.uses_q12_length());
        assert!(LogikVersion::V3EarlyVol64.has_vol64_loop_quirk());
        assert!(!LogikVersion::V3EarlyVol64.has_ima_adpcm_mixer());

        assert!(LogikVersion::V3EarlyVol64Adpcm.has_ima_adpcm_mixer());
        assert_eq!(LogikVersion::V3EarlyVol64Adpcm.pcm_mixer_size_bytes(), 0x120);

        assert!(LogikVersion::V3Standard.uses_q12_length());
        assert!(!LogikVersion::V3Standard.has_vol64_fast_path());
        assert!(!LogikVersion::V3Standard.has_vol64_loop_quirk());
        assert_eq!(LogikVersion::V3Standard.pcm_mixer_size_bytes(), 0xA0);
    }
}
