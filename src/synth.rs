//! Dual synthesis pipelines for Logik State GBAModPlay:
//!
//! - [`AccuracySynth`]: 100% bit-exact emulation of all **6** Logik State ARM7TDMI mixer
//!   revisions (`V1Mod1`, `V2Mod2Halfword`, `V2Mod2Word`, `V3EarlyVol64`, `V3EarlyVol64Adpcm`,
//!   and `V3Standard`), including the dedicated 2x-upsampled high-nibble-first 4-bit IMA ADPCM
//!   ARM mixer.
//! - [`StudioSynth`]: 48,000 Hz `f32` band-limited Lanczos-3 polyphase windowed-sinc renderer
//!   (with general Lanczos-`a` fallback) and true equal-power stereo panning.

use crate::freq::{IMA_INDEX_TABLE, IMA_STEP_TABLE};
use crate::sample::LogikSampleView;
use crate::version::LogikVersion;
use crate::voice::{LogikAdpcmVoice, LogikPcmVoice};

/// Bit-exact software emulation of all **6** Logik State GBAModPlay ARM7TDMI mixer revisions.
///
/// Every mixing kernel in `AccuracySynth` is verified strictly byte-for-byte (`assert_eq!`,
/// zero tolerance) against real ARM7TDMI execution captured in headless `libmgba.so.0.10`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AccuracySynth;

impl AccuracySynth {
    /// Mix a **GBAModPlay v1.x** ([`LogikVersion::V1Mod1`], *Pinball Challenge Deluxe* `APLP`)
    /// PCM8 voice into an 8-bit DirectSound mixing buffer using wrapping 8-bit addition.
    ///
    /// Exact ARM7TDMI semantics:
    /// - 16-bit halfword `length`, `loop_start`, `loop_len`.
    /// - Unsigned `lsr r6, r4, #12` index extraction.
    /// - On `r6 >= length`: `ptr += loop_start`, `loop_start = 0`, `length = loop_len`,
    ///   `phase &= 0x0FFF`, `r6 = 0`.
    /// - `out[i] = out[i].wrapping_add(((sample * volume) as u32 >> 8) as i8)`.
    pub fn mix_v1_mod1(pcm_bytes: &[u8], voice: &mut LogikPcmVoice, out: &mut [i8]) {
        if !voice.active || pcm_bytes.is_empty() || out.is_empty() {
            return;
        }
        let mut ptr = voice.ptr_offset;
        let mut phase = voice.phase_q12;
        let step = voice.step_q12;
        let mut sl = voice.length & 0xFFFF;
        let mut ip = voice.loop_start & 0xFFFF;
        let r0 = voice.loop_len & 0xFFFF;
        let vol = voice.volume as i32;
        let pcm_len = pcm_bytes.len();

        for dst in out.iter_mut() {
            let mut r6 = phase >> 12;
            if (r6 as i32) >= (sl as i32) {
                ptr = ptr.wrapping_add(ip as usize);
                ip = 0;
                sl = r0;
                phase &= 0x0FFF;
                r6 = phase >> 12;
            }
            phase = phase.wrapping_add(step);
            let idx = ptr.wrapping_add(r6 as usize);
            if idx >= pcm_len {
                voice.active = false;
                break;
            }
            let s = pcm_bytes[idx] as i8 as i32;
            let prod = s.wrapping_mul(vol);
            *dst = dst.wrapping_add(((prod as u32) >> 8) as i8);
        }

        voice.ptr_offset = ptr;
        voice.phase_q12 = phase;
        voice.length = sl;
        voice.loop_start = ip;
    }

    /// Shared implementation for **GBAModPlay v2.x** integer-length mixers
    /// ([`LogikVersion::V2Mod2Halfword`] and [`LogikVersion::V2Mod2Word`]).
    #[inline(always)]
    fn mix_v2_impl<const HALFWORD: bool>(
        pcm_bytes: &[u8],
        voice: &mut LogikPcmVoice,
        out: &mut [i8],
    ) {
        if !voice.active || pcm_bytes.is_empty() || out.is_empty() {
            return;
        }
        let mut sl: u32 = if HALFWORD {
            voice.length & 0xFFFF
        } else {
            voice.length
        };
        // Exact ARM7TDMI guard: `cmp sl, #2; beq done` (returns without writing back)
        if sl == 2 {
            return;
        }

        let mut ptr = voice.ptr_offset;
        let mut phase = voice.phase_q12;
        let step = voice.step_q12;
        let mut ip: u32 = if HALFWORD {
            voice.loop_start & 0xFFFF
        } else {
            voice.loop_start
        };
        let r0: u32 = if HALFWORD {
            voice.loop_len & 0xFFFF
        } else {
            voice.loop_len
        };
        let vol = voice.volume as i32;
        let vol64 = voice.volume == 0x40;
        let pcm_len = pcm_bytes.len();

        for dst in out.iter_mut() {
            let mut r6 = (phase as i32) >> 12;
            if r6 >= (sl as i32) {
                ptr = ptr.wrapping_add(ip as usize);
                ip = 0;
                sl = r0;
                phase &= 0x0FFF;
                r6 = (phase as i32) >> 12;
            }
            phase = phase.wrapping_add(step);
            let idx = (ptr as isize).wrapping_add(r6 as isize) as usize;
            if idx >= pcm_len {
                voice.active = false;
                break;
            }
            let s = pcm_bytes[idx] as i8 as i32;
            let scaled = if vol64 {
                s >> 2
            } else {
                s.wrapping_mul(vol) >> 8
            };
            *dst = dst.wrapping_add(scaled as i8);
        }

        voice.ptr_offset = ptr;
        voice.phase_q12 = phase;
        voice.length = sl;
        voice.loop_start = ip;
    }

    /// Mix a **GBAModPlay v2.x halfword** ([`LogikVersion::V2Mod2Halfword`], *Aero the Acro-Bat*)
    /// PCM8 voice into an 8-bit DirectSound mixing buffer.
    #[inline]
    pub fn mix_v2_halfword(pcm_bytes: &[u8], voice: &mut LogikPcmVoice, out: &mut [i8]) {
        Self::mix_v2_impl::<true>(pcm_bytes, voice, out);
    }

    /// Mix a **GBAModPlay v2.x word** ([`LogikVersion::V2Mod2Word`], *Zapper*) PCM8 voice
    /// into an 8-bit DirectSound mixing buffer.
    #[inline]
    pub fn mix_v2_word(pcm_bytes: &[u8], voice: &mut LogikPcmVoice, out: &mut [i8]) {
        Self::mix_v2_impl::<false>(pcm_bytes, voice, out);
    }

    /// Mix a **GBAModPlay early v3.0** ([`LogikVersion::V3EarlyVol64`] /
    /// [`LogikVersion::V3EarlyVol64Adpcm`], *Stuntman*, *Super Puzzle Fighter II*) PCM8 voice.
    ///
    /// Reproduces the exact `vol == 64` (`0x40`) fast-path ARM loop, including the authentic
    /// hardware quirk where the first sample immediately after a loop wrap reuses the pre-wrap
    /// `r6 = phase >> 12` index because the loop handler branches past `asr r6, r4, #12`.
    pub fn mix_v3_early_vol64(pcm_bytes: &[u8], voice: &mut LogikPcmVoice, out: &mut [i8]) {
        if !voice.active || pcm_bytes.is_empty() || out.is_empty() {
            return;
        }
        let mut sl = voice.length;
        // Exact ARM7TDMI guard: `tst sl, sl; beq done`
        if sl == 0 {
            voice.active = false;
            return;
        }

        if voice.volume != 0x40 {
            Self::mix_v3_standard(pcm_bytes, voice, out);
            return;
        }

        let mut ptr = voice.ptr_offset;
        let mut phase = voice.phase_q12;
        let step = voice.step_q12;
        let mut ip = voice.loop_start;
        let r0 = voice.loop_len;
        let pcm_len = pcm_bytes.len();

        for dst in out.iter_mut() {
            // Exact ARM7TDMI order in `vol == 0x40` fast path:
            //   +0x0C0: asr r6, r4, #0xc
            //   +0x0C4: cmp r4, sl
            //   +0x0C8: bge loop_wrap   (loop_wrap branches back to +0x0CC, reusing `r6`!)
            let r6 = (phase as i32) >> 12;
            if (phase as i32) >= (sl as i32) {
                ptr = ptr.wrapping_add(ip as usize);
                ip = 0;
                phase = phase.wrapping_sub(sl);
                sl = r0;
                if sl == 0 {
                    voice.active = false;
                    break;
                }
            }
            phase = phase.wrapping_add(step);
            let idx = (ptr as isize).wrapping_add(r6 as isize) as usize;
            if idx >= pcm_len {
                voice.active = false;
                break;
            }
            let s = pcm_bytes[idx] as i8 as i32;
            *dst = dst.wrapping_add((s >> 2) as i8);
        }

        voice.ptr_offset = ptr;
        voice.phase_q12 = phase;
        voice.length = sl;
        voice.loop_start = ip;
    }

    /// Mix a **GBAModPlay standard v3.0b / v3.1** ([`LogikVersion::V3Standard`], 47 ROMs)
    /// PCM8 voice into an 8-bit DirectSound mixing buffer.
    ///
    /// Exact `0xA0`-byte ARM7TDMI mixer:
    /// - `tst sl, sl; beq done`
    /// - `cmp r4, sl; bge loop_wrap` (Q20.12 signed comparison)
    /// - Loop wrap: `ptr += loop_start`, `loop_start = 0`, `phase -= sl`, `sl = loop_len`;
    ///   if `sl == 0` -> stop voice (one-shot).
    /// - `r6 = phase >> 12; phase += step; out[i] += ((pcm[ptr + r6] * vol) >> 8) as i8`.
    pub fn mix_v3_standard(pcm_bytes: &[u8], voice: &mut LogikPcmVoice, out: &mut [i8]) {
        if !voice.active || pcm_bytes.is_empty() || out.is_empty() {
            return;
        }
        let mut sl = voice.length;
        if sl == 0 {
            voice.active = false;
            return;
        }

        let mut ptr = voice.ptr_offset;
        let mut phase = voice.phase_q12;
        let step = voice.step_q12;
        let mut ip = voice.loop_start;
        let r0 = voice.loop_len;
        let vol = voice.volume as i32;
        let pcm_len = pcm_bytes.len();

        for dst in out.iter_mut() {
            if (phase as i32) >= (sl as i32) {
                ptr = ptr.wrapping_add(ip as usize);
                ip = 0;
                phase = phase.wrapping_sub(sl);
                sl = r0;
                if sl == 0 {
                    voice.active = false;
                    break;
                }
            }
            let r6 = (phase as i32) >> 12;
            phase = phase.wrapping_add(step);
            let idx = (ptr as isize).wrapping_add(r6 as isize) as usize;
            if idx >= pcm_len {
                voice.active = false;
                break;
            }
            let s = pcm_bytes[idx] as i8 as i32;
            let scaled = s.wrapping_mul(vol) >> 8;
            *dst = dst.wrapping_add(scaled as i8);
        }

        voice.ptr_offset = ptr;
        voice.phase_q12 = phase;
        voice.length = sl;
        voice.loop_start = ip;
    }

    /// Dispatch a PCM8 voice using the selected [`LogikVersion`] ARM7TDMI mixer revision.
    #[inline]
    pub fn mix_pcm_voice(
        version: LogikVersion,
        pcm_bytes: &[u8],
        voice: &mut LogikPcmVoice,
        out: &mut [i8],
    ) {
        match version {
            LogikVersion::V1Mod1 => Self::mix_v1_mod1(pcm_bytes, voice, out),
            LogikVersion::V2Mod2Halfword => Self::mix_v2_halfword(pcm_bytes, voice, out),
            LogikVersion::V2Mod2Word => Self::mix_v2_word(pcm_bytes, voice, out),
            LogikVersion::V3EarlyVol64 | LogikVersion::V3EarlyVol64Adpcm => {
                Self::mix_v3_early_vol64(pcm_bytes, voice, out)
            }
            LogikVersion::V3Standard => Self::mix_v3_standard(pcm_bytes, voice, out),
        }
    }

    /// Mix a **4-bit IMA ADPCM** voice using the exact **Logik State `V3EarlyVol64Adpcm`**
    /// ARM7TDMI mixer (*Super Puzzle Fighter II*, `AZ8E` / `AZ8P`).
    ///
    /// Exact ARM7TDMI semantics:
    /// - High nibble first when `nibble_toggle == 0`, low nibble when non-zero; `nibble_toggle = !nibble_toggle`.
    /// - `step_index = (step_index + IMA_INDEX_TABLE[nib]).clamp(0, 88)`.
    /// - `diff = (((nib & 7) * step) as u32) >> 2` (using the pre-update `step`, no `+ step >> 3`).
    /// - `predictor = if (nib & 8) != 0 { predictor - diff } else { predictor + diff }` (unclamped `i32`).
    /// - Reload `step = IMA_STEP_TABLE[step_index]`.
    /// - `scaled = ((predictor * volume) as u32 >> 16) as i8`.
    /// - **2x sample-hold upsampling**: mixes `scaled` into `out[i]` AND `out[i + 1]` before
    ///   decrementing `remaining_nibbles`.
    pub fn mix_v3_adpcm_voice(
        adpcm_bytes: &[u8],
        voice: &mut LogikAdpcmVoice,
        out: &mut [i8],
    ) {
        if !voice.active || adpcm_bytes.is_empty() || out.is_empty() {
            return;
        }

        let vol = voice.volume as i32;
        let mut pred = voice.predictor;
        let mut step_idx = voice.step_index.clamp(0, 88);
        let mut step = IMA_STEP_TABLE[step_idx as usize];
        let mut byte_off = voice.byte_offset;
        let mut toggle = voice.nibble_toggle;
        let mut cached = voice.cached_byte;
        let mut rem = voice.remaining_nibbles;
        let mut out_idx = 0usize;
        let out_len = out.len();
        let data_len = adpcm_bytes.len();

        while out_idx < out_len {
            let nib = if toggle == 0 {
                if byte_off >= data_len {
                    voice.active = false;
                    break;
                }
                cached = adpcm_bytes[byte_off];
                byte_off += 1;
                (cached >> 4) & 0x0F
            } else {
                cached & 0x0F
            };
            toggle = !toggle;

            step_idx = (step_idx + IMA_INDEX_TABLE[nib as usize]).clamp(0, 88);
            let diff = ((((nib & 7) as i32).wrapping_mul(step)) as u32 >> 2) as i32;
            if (nib & 8) != 0 {
                pred = pred.wrapping_sub(diff);
            } else {
                pred = pred.wrapping_add(diff);
            }
            step = IMA_STEP_TABLE[step_idx as usize];

            let scaled = ((pred.wrapping_mul(vol) as u32) >> 16) as i8;

            // Sample #1
            out[out_idx] = out[out_idx].wrapping_add(scaled);
            out_idx += 1;
            if out_idx >= out_len {
                break;
            }

            // Sample #2 (2x sample-hold doubling)
            out[out_idx] = out[out_idx].wrapping_add(scaled);
            out_idx += 1;
            if out_idx >= out_len {
                break;
            }

            rem = rem.wrapping_sub(1);
            if rem == 0 {
                voice.active = false;
                break;
            }
        }

        voice.remaining_nibbles = rem;
        voice.predictor = pred;
        voice.step_index = step_idx;
        voice.byte_offset = byte_off;
        voice.nibble_toggle = toggle;
        voice.cached_byte = cached;
    }

    /// Convenience stereo helper: mixes a PCM8 voice into separate DirectSound A (left) and
    /// DirectSound B (right) buffers using independent left/right channel volumes while advancing
    /// `voice` once.
    pub fn mix_pcm_stereo_voice(
        version: LogikVersion,
        pcm_bytes: &[u8],
        voice: &mut LogikPcmVoice,
        vol_left: u32,
        vol_right: u32,
        out_left: &mut [i8],
        out_right: &mut [i8],
    ) {
        let frames = out_left.len().min(out_right.len());
        if frames == 0 || !voice.active {
            return;
        }
        let mut right_state = *voice;
        voice.volume = vol_left;
        right_state.volume = vol_right;

        Self::mix_pcm_voice(version, pcm_bytes, voice, &mut out_left[..frames]);
        Self::mix_pcm_voice(version, pcm_bytes, &mut right_state, &mut out_right[..frames]);
    }
}

// ============================================================================
// StudioSynth: 48,000 Hz f32 Band-Limited Lanczos-3 + Equal-Power Panning
// ============================================================================

/// Number of fractional polyphase steps in [`StudioSynth`]'s compile-time Lanczos-3 LUT.
const LANCZOS_PHASES: usize = 1024;

const fn sin_const(mut x: f64) -> f64 {
    const PI: f64 = core::f64::consts::PI;
    const TWO_PI: f64 = core::f64::consts::TAU;
    const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;
    let k = (x / TWO_PI) as i64;
    x -= (k as f64) * TWO_PI;
    if x > PI {
        x -= TWO_PI;
    } else if x < -PI {
        x += TWO_PI;
    }
    if x > HALF_PI {
        x = PI - x;
    } else if x < -HALF_PI {
        x = -PI - x;
    }
    let r2 = x * x;
    x * (1.0
        - r2 * (1.0 / 6.0
            - r2 * (1.0 / 120.0
                - r2 * (1.0 / 5_040.0
                    - r2 * (1.0 / 362_880.0
                        - r2 * (1.0 / 39_916_800.0 - r2 / 6_227_020_800.0))))))
}

const fn lanczos_kernel(x: f64, a: f64) -> f64 {
    let ax = if x < 0.0 { -x } else { x };
    if ax < 1e-12 {
        return 1.0;
    }
    if ax >= a {
        return 0.0;
    }
    let pi_x = core::f64::consts::PI * x;
    let pi_x_a = pi_x / a;
    (sin_const(pi_x) / pi_x) * (sin_const(pi_x_a) / pi_x_a)
}

const fn build_lanczos3_lut() -> [[f32; 6]; LANCZOS_PHASES + 1] {
    let mut lut = [[0.0f32; 6]; LANCZOS_PHASES + 1];
    let mut p = 0usize;
    while p <= LANCZOS_PHASES {
        let frac = (p as f64) / (LANCZOS_PHASES as f64);
        let mut weights = [0.0f64; 6];
        let mut sum = 0.0f64;
        let mut t = 0usize;
        while t < 6 {
            let k = (t as i32) - 2;
            let w = lanczos_kernel(frac - (k as f64), 3.0);
            weights[t] = w;
            sum += w;
            t += 1;
        }
        let inv_sum = if sum > 1e-12 { 1.0 / sum } else { 0.0 };
        let mut j = 0usize;
        while j < 6 {
            lut[p][j] = (weights[j] * inv_sum) as f32;
            j += 1;
        }
        p += 1;
    }
    lut
}

static LANCZOS3_LUT: [[f32; 6]; LANCZOS_PHASES + 1] = build_lanczos3_lut();

#[inline]
fn cos_f64(x: f64) -> f64 {
    sin_const(core::f64::consts::FRAC_PI_2 - x)
}

#[inline]
fn sin_f64(x: f64) -> f64 {
    sin_const(x)
}

/// High-fidelity 48,000 Hz `f32` software synthesizer with band-limited Lanczos-3 polyphase
/// windowed-sinc interpolation (and general Lanczos-`a` fallback) plus equal-power stereo panning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StudioSynth {
    /// Output sample rate in Hz (default [`Self::SAMPLE_RATE`] = `48_000`).
    pub sample_rate: u32,
    /// Number of Lanczos kernel lobes (default `3`, using the compile-time 1,024-phase 6-tap LUT).
    pub sinc_lobes: usize,
}

impl Default for StudioSynth {
    fn default() -> Self {
        Self::new(Self::SAMPLE_RATE)
    }
}

impl StudioSynth {
    /// Standard studio output sample rate (`48_000` Hz).
    pub const SAMPLE_RATE: u32 = 48_000;

    /// Create a new `StudioSynth` at the specified output sample rate (typically `48_000` Hz).
    #[must_use]
    pub const fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            sinc_lobes: 3,
        }
    }

    /// Render a [`LogikSampleView`] into stereo `f32` buffers (accumulating) using band-limited
    /// Lanczos windowed-sinc interpolation and true equal-power stereo panning.
    ///
    /// - `freq_hz`: desired playback sample rate in Hz.
    /// - `volume`: linear gain in `0.0..=1.0`.
    /// - `pan`: stereo pan in `-1.0..=1.0` (`-1.0` = hard left, `0.0` = center, `+1.0` = hard right).
    #[allow(clippy::too_many_arguments)]
    pub fn render_voice_f32(
        &self,
        sample: &LogikSampleView<'_>,
        pos_f64: &mut f64,
        freq_hz: f64,
        volume: f32,
        pan: f32,
        out_left: &mut [f32],
        out_right: &mut [f32],
    ) {
        let frames = out_left.len().min(out_right.len());
        if frames == 0
            || sample.is_empty()
            || self.sample_rate == 0
            || !freq_hz.is_finite()
            || freq_hz <= 0.0
            || !pos_f64.is_finite()
        {
            return;
        }
        if *pos_f64 < 0.0 {
            *pos_f64 = 0.0;
        }

        let pcm = sample.pcm_bytes;
        let pcm_len = pcm.len();
        let step = freq_hz / (self.sample_rate as f64);

        let pan_clamped = pan.clamp(-1.0, 1.0) as f64;
        let angle = (pan_clamped + 1.0) * core::f64::consts::FRAC_PI_4;
        let gain_l = (cos_f64(angle) as f32) * volume * (1.0 / 128.0);
        let gain_r = (sin_f64(angle) as f32) * volume * (1.0 / 128.0);

        let loop_bounds = sample.loop_start.zip(sample.loop_end).and_then(|(ls, le)| {
            let ls = ls as usize;
            let le = (le as usize).min(pcm_len);
            if le > ls { Some((ls, le)) } else { None }
        });

        let lobes = self.sinc_lobes.max(1);
        let mut pos = *pos_f64;

        let fetch = |idx: isize| -> f32 {
            if let Some((ls, le)) = loop_bounds {
                let loop_len = (le - ls) as isize;
                let wrapped = if idx < 0 {
                    0usize
                } else if idx >= le as isize {
                    let over = (idx - ls as isize).rem_euclid(loop_len);
                    ls + (over as usize)
                } else {
                    idx as usize
                };
                (pcm[wrapped] as i8) as f32
            } else if idx < 0 || idx >= pcm_len as isize {
                0.0
            } else {
                (pcm[idx as usize] as i8) as f32
            }
        };

        for i in 0..frames {
            if let Some((ls, le)) = loop_bounds {
                let le_f = le as f64;
                let loop_len_f = (le - ls) as f64;
                if pos >= le_f {
                    pos = (ls as f64) + ((pos - (ls as f64)) % loop_len_f);
                }
            } else if pos >= pcm_len as f64 {
                break;
            }

            let base = pos as isize;
            let frac = pos - (base as f64);

            let interpolated = if lobes == 3 {
                let p_idx = ((frac * (LANCZOS_PHASES as f64) + 0.5) as usize).min(LANCZOS_PHASES);
                let w = &LANCZOS3_LUT[p_idx];
                w[0] * fetch(base - 2)
                    + w[1] * fetch(base - 1)
                    + w[2] * fetch(base)
                    + w[3] * fetch(base + 1)
                    + w[4] * fetch(base + 2)
                    + w[5] * fetch(base + 3)
            } else {
                let a = lobes as isize;
                let a_f = lobes as f64;
                let mut acc = 0.0f64;
                let mut wsum = 0.0f64;
                let mut k = 1 - a;
                while k <= a {
                    let w = lanczos_kernel(frac - (k as f64), a_f);
                    acc += w * (fetch(base + k) as f64);
                    wsum += w;
                    k += 1;
                }
                if wsum > 1e-12 {
                    (acc / wsum) as f32
                } else {
                    0.0
                }
            };

            out_left[i] += interpolated * gain_l;
            out_right[i] += interpolated * gain_r;
            pos += step;
        }

        *pos_f64 = pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_studio_synth_equal_power_panning_and_fallback() {
        let pcm = [64i8 as u8; 64];
        let view = LogikSampleView::from_pcm_bytes(10_512, 64, Some(0), Some(64), &pcm);
        let studio = StudioSynth::default();

        let mut l = [0.0f32; 64];
        let mut r = [0.0f32; 64];
        let mut pos = 0.0;
        studio.render_voice_f32(&view, &mut pos, 10_512.0, 1.0, 0.0, &mut l, &mut r);
        assert!((l[32] - r[32]).abs() < 1e-5);
        assert!(l[32] > 0.3);

        // General Lanczos-a fallback (lobes = 4)
        let studio4 = StudioSynth {
            sinc_lobes: 4,
            ..Default::default()
        };
        let mut l4 = [0.0f32; 32];
        let mut r4 = [0.0f32; 32];
        let mut pos4 = 0.0;
        studio4.render_voice_f32(&view, &mut pos4, 10_512.0, 1.0, -1.0, &mut l4, &mut r4);
        assert!(l4[16] > 0.4);
        assert!(r4[16].abs() < 1e-5);
    }
}
