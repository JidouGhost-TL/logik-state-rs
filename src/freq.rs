//! GBA hardware timing, Q20.12 fixed-point pitch conversion, and IMA ADPCM tables
//! for Logik State GBAModPlay.

/// Native Game Boy Advance master clock frequency (`16,777,216` Hz = `2^24` Hz).
pub const GBA_CLOCK_HZ: u32 = 16_777_216;

/// Unity (`1.0x`) Q20.12 fixed-point phase increment (`4,096`).
pub const Q12_ONE: u32 = 0x1000;

/// Default GBAModPlay hardware DirectSound mixing rate (`10,512` Hz, Timer0 reload `0xF9C4`).
pub const LOGIK_DEFAULT_MIX_RATE_HZ: u32 = 10_512;

/// Standard Timer0 reload table found in Logik State GBAModPlay v3.0 builds
/// (e.g. *Stuntman*, *Need for Speed: Porsche Unleashed*, *Medal of Honor: Infiltrator*):
/// `10,512`, `13,336`, `15,280`, `18,001`, `21,024`, `26,715`, `31,477`, `36,236` Hz.
pub const LOGIK_TIMER0_RELOADS_V3: [u16; 8] = [
    0xF9C4, 0xFB16, 0xFBB6, 0xFC5C, 0xFCE2, 0xFD8C, 0xFDEB, 0xFE31,
];

/// Alternate Timer0 reload table found in 2002 *Super Puzzle Fighter II* (`AZ8E`/`AZ8P`) builds:
/// `10,505`, `13,368`, `15,280`, `18,138`, `21,024`, `26,715`, `31,477`, `36,236` Hz.
pub const LOGIK_TIMER0_RELOADS_V2_ADPCM: [u16; 8] = [
    0xF9C3, 0xFB19, 0xFBB6, 0xFC63, 0xFCE2, 0xFD8C, 0xFDEB, 0xFE31,
];

/// Alternate Timer0 reload table found in 2003+ GBAModPlay v3.0b+ builds
/// (e.g. *Alien Hominid*, *Bionicle*, *Need for Speed: Underground 2*):
/// uses `0xFBB7` (`15,294` Hz) in slot 2 instead of `0xFBB6`.
pub const LOGIK_TIMER0_RELOADS_V3B: [u16; 8] = [
    0xF9C4, 0xFB16, 0xFBB7, 0xFC5C, 0xFCE2, 0xFD8C, 0xFDEB, 0xFE31,
];

/// Standard 89-entry IMA ADPCM step-size lookup table embedded in Logik State's
/// `V3EarlyVol64Adpcm` ARM7TDMI mixer (`AZ8E` / `AZ8P`).
pub const IMA_STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

/// Standard 16-entry IMA ADPCM step-index adjustment table embedded immediately before
/// [`IMA_STEP_TABLE`] in Logik State's `V3EarlyVol64Adpcm` ARM7TDMI mixer.
pub const IMA_INDEX_TABLE: [i32; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

/// Compute the hardware DirectSound sample rate in Hz from a GBA Timer0 reload value.
///
/// Formula: `round(16_777_216 / (65536 - reload))`.
#[inline]
#[must_use]
pub const fn rate_from_timer_reload(reload: u16) -> u32 {
    let ticks = 65_536u32 - (reload as u32);
    (GBA_CLOCK_HZ + (ticks / 2)) / ticks
}

/// Convert a desired playback frequency in Hz into a Q20.12 fixed-point phase step
/// relative to `mix_rate_hz`.
#[inline]
#[must_use]
pub const fn freq_to_step_q12(freq_hz: u32, mix_rate_hz: u32) -> u32 {
    if mix_rate_hz == 0 {
        return 0;
    }
    let num = ((freq_hz as u64) << 12) + ((mix_rate_hz as u64) / 2);
    let step = num / (mix_rate_hz as u64);
    if step > u32::MAX as u64 {
        u32::MAX
    } else {
        step as u32
    }
}

/// Convert a Q20.12 fixed-point phase step back into a playback frequency in Hz
/// relative to `mix_rate_hz`.
#[inline]
#[must_use]
pub const fn step_q12_to_freq(step_q12: u32, mix_rate_hz: u32) -> f64 {
    ((step_q12 as f64) * (mix_rate_hz as f64)) / (Q12_ONE as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_reload_rates() {
        assert_eq!(rate_from_timer_reload(0xF9C4), 10_512);
        assert_eq!(rate_from_timer_reload(0xFB16), 13_336);
        assert_eq!(rate_from_timer_reload(0xFBB6), 15_280);
        assert_eq!(rate_from_timer_reload(0xFC5C), 18_001);
        assert_eq!(rate_from_timer_reload(0xFCE2), 21_024);
        assert_eq!(rate_from_timer_reload(0xFD8C), 26_715);
        assert_eq!(rate_from_timer_reload(0xFDEB), 31_477);
        assert_eq!(rate_from_timer_reload(0xFE31), 36_236);
    }

    #[test]
    fn test_q12_roundtrip() {
        let step = freq_to_step_q12(10_512, 10_512);
        assert_eq!(step, Q12_ONE);
        assert!((step_q12_to_freq(step, 10_512) - 10_512.0).abs() < 1e-6);
        assert_eq!(freq_to_step_q12(21_024, 10_512), Q12_ONE * 2);
        assert_eq!(freq_to_step_q12(10_512, 0), 0);
    }
}
