//! Dual synthesis pipelines for Logik State GBAModPlay:
//!
//! - [`AccuracySynth`]: 100% bit-exact emulation of all **6** Logik State ARM7TDMI mixer
//!   revisions (`V1Mod1`, `V2Mod2Halfword`, `V2Mod2Word`, `V3EarlyVol64`, `V3EarlyVol64Adpcm`,
//!   and `V3Standard`), including the dedicated 2x-upsampled high-nibble-first 4-bit IMA ADPCM
//!   ARM mixer.
//! - [`StudioSynth`]: 48,000 Hz `f32` band-limited Lanczos-3 polyphase windowed-sinc renderer
//!   (with general Lanczos-`a` fallback) and true equal-power stereo panning.

use crate::freq::IMA_STEP_TABLE;
use crate::sample::LogikSampleView;
use crate::version::LogikVersion;
use crate::voice::{LogikAdpcmVoice, LogikPcmVoice};

/// Compile-time 89 × 16 signed predictor-delta LUT for Logik State `V3EarlyVol64Adpcm`:
/// `±(((nib & 7) * IMA_STEP_TABLE[idx]) >> 2)`.
///
/// Keeping `step_idx` updated via register arithmetic (`ima_index_delta` + `clamp(0, 88)`)
/// breaks the L1 load-to-use latency chain on `step_idx`, while this table removes the
/// per-nibble multiply, shift, and sign branch.
const fn build_ima_signed_diff_lut() -> [[i32; 16]; 89] {
    let mut table = [[0i32; 16]; 89];
    let mut idx = 0usize;
    while idx < 89 {
        let step = IMA_STEP_TABLE[idx];
        let mut nib = 0usize;
        while nib < 16 {
            let diff = ((((nib & 7) as i32) * step) as u32 >> 2) as i32;
            table[idx][nib] = if (nib & 8) != 0 { -diff } else { diff };
            nib += 1;
        }
        idx += 1;
    }
    table
}

static IMA_SIGNED_DIFF_LUT: [[i32; 16]; 89] = build_ima_signed_diff_lut();

#[inline(always)]
const fn ima_index_delta(nib: u8) -> i32 {
    let mag = (nib & 7) as i32;
    if mag < 4 {
        -1
    } else {
        (mag - 3) << 1
    }
}

/// Fast interior-run PCM8 mixer kernel (no loop wrap, no out-of-bounds).
///
/// Caller must guarantee that for all `k` in `0..dst.len()`:
/// - `(phase0 as u64) + (k as u64) * (step as u64) < (1u64 << 32)` (and `< (1u64 << 31)` when
///   signed shift semantics apply), and
/// - `(((phase0 as u64) + (k as u64) * (step as u64)) >> 12) < src.len()`.
#[inline(always)]
fn mix_pcm_interior_run<const VOL64: bool>(
    src: &[u8],
    dst: &mut [i8],
    phase0: u32,
    step: u32,
    vol: i32,
) {
    if step == 0x1000 {
        let start = (phase0 >> 12) as usize;
        let s_slice = &src[start..start + dst.len()];
        if VOL64 {
            for (&b, d) in s_slice.iter().zip(dst.iter_mut()) {
                *d = d.wrapping_add((b as i8) >> 2);
            }
        } else if (0..=255).contains(&vol) {
            let v16 = vol as i16;
            for (&b, d) in s_slice.iter().zip(dst.iter_mut()) {
                let s = b as i8 as i16;
                *d = d.wrapping_add(((s * v16) >> 8) as i8);
            }
        } else {
            for (&b, d) in s_slice.iter().zip(dst.iter_mut()) {
                let s = b as i8 as i32;
                *d = d.wrapping_add((s.wrapping_mul(vol) >> 8) as i8);
            }
        }
        return;
    }

    let mut ph = phase0;
    let step2 = step.wrapping_mul(2);
    let step3 = step.wrapping_mul(3);
    let step4 = step.wrapping_mul(4);
    let step5 = step.wrapping_mul(5);
    let step6 = step.wrapping_mul(6);
    let step7 = step.wrapping_mul(7);
    let step8 = step.wrapping_mul(8);

    let (chunks, remainder) = dst.as_chunks_mut::<8>();
    if VOL64 {
        for chunk in chunks {
            let s = [
                src[(ph >> 12) as usize] as i8,
                src[(ph.wrapping_add(step) >> 12) as usize] as i8,
                src[(ph.wrapping_add(step2) >> 12) as usize] as i8,
                src[(ph.wrapping_add(step3) >> 12) as usize] as i8,
                src[(ph.wrapping_add(step4) >> 12) as usize] as i8,
                src[(ph.wrapping_add(step5) >> 12) as usize] as i8,
                src[(ph.wrapping_add(step6) >> 12) as usize] as i8,
                src[(ph.wrapping_add(step7) >> 12) as usize] as i8,
            ];
            ph = ph.wrapping_add(step8);
            for j in 0..8 {
                chunk[j] = chunk[j].wrapping_add(s[j] >> 2);
            }
        }
    } else if (0..=255).contains(&vol) {
        let v16 = vol as i16;
        for chunk in chunks {
            let s = [
                src[(ph >> 12) as usize] as i8 as i16,
                src[(ph.wrapping_add(step) >> 12) as usize] as i8 as i16,
                src[(ph.wrapping_add(step2) >> 12) as usize] as i8 as i16,
                src[(ph.wrapping_add(step3) >> 12) as usize] as i8 as i16,
                src[(ph.wrapping_add(step4) >> 12) as usize] as i8 as i16,
                src[(ph.wrapping_add(step5) >> 12) as usize] as i8 as i16,
                src[(ph.wrapping_add(step6) >> 12) as usize] as i8 as i16,
                src[(ph.wrapping_add(step7) >> 12) as usize] as i8 as i16,
            ];
            ph = ph.wrapping_add(step8);
            for j in 0..8 {
                chunk[j] = chunk[j].wrapping_add(((s[j] * v16) >> 8) as i8);
            }
        }
    } else {
        for chunk in chunks {
            let s = [
                src[(ph >> 12) as usize] as i8 as i32,
                src[(ph.wrapping_add(step) >> 12) as usize] as i8 as i32,
                src[(ph.wrapping_add(step2) >> 12) as usize] as i8 as i32,
                src[(ph.wrapping_add(step3) >> 12) as usize] as i8 as i32,
                src[(ph.wrapping_add(step4) >> 12) as usize] as i8 as i32,
                src[(ph.wrapping_add(step5) >> 12) as usize] as i8 as i32,
                src[(ph.wrapping_add(step6) >> 12) as usize] as i8 as i32,
                src[(ph.wrapping_add(step7) >> 12) as usize] as i8 as i32,
            ];
            ph = ph.wrapping_add(step8);
            for j in 0..8 {
                chunk[j] = chunk[j].wrapping_add((s[j].wrapping_mul(vol) >> 8) as i8);
            }
        }
    }

    for d in remainder {
        let s = src[(ph >> 12) as usize] as i8 as i32;
        ph = ph.wrapping_add(step);
        let scaled = if VOL64 {
            s >> 2
        } else {
            s.wrapping_mul(vol) >> 8
        };
        *d = d.wrapping_add(scaled as i8);
    }
}

/// Bit-exact software emulation of all **6** Logik State GBAModPlay ARM7TDMI mixer revisions.
///
/// Every mixing kernel in `AccuracySynth` is verified strictly byte-for-byte (`assert_eq!`,
/// zero tolerance) against real ARM7TDMI execution captured in headless `libmgba.so.0.10`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
        let out_len = out.len();
        let mut offset = 0usize;

        while offset < out_len {
            if step > 0 && sl > 0 && ptr < pcm_len {
                let safe_samples = (sl as usize).min(pcm_len - ptr);
                let limit_phase = ((safe_samples as u64) << 12).min(0x1_0000_0000u64);
                if (phase as u64) < limit_phase {
                    let k_max = (limit_phase - 1 - (phase as u64)) / (step as u64);
                    let run = ((k_max + 1) as usize).min(out_len - offset);
                    mix_pcm_interior_run::<false>(
                        &pcm_bytes[ptr..pcm_len],
                        &mut out[offset..offset + run],
                        phase,
                        step,
                        vol,
                    );
                    phase = phase.wrapping_add(step.wrapping_mul(run as u32));
                    offset += run;
                    continue;
                }
            }

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
            out[offset] = out[offset].wrapping_add(((prod as u32) >> 8) as i8);
            offset += 1;
        }

        voice.ptr_offset = ptr;
        voice.phase_q12 = phase;
        voice.length = sl;
        voice.loop_start = ip;
    }

    /// Shared implementation for **GBAModPlay v2.x** integer-length mixers
    /// ([`LogikVersion::V2Mod2Halfword`] and [`LogikVersion::V2Mod2Word`]).
    #[inline(always)]
    fn mix_v2_impl<const HALFWORD: bool, const VOL64: bool>(
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
        let pcm_len = pcm_bytes.len();
        let out_len = out.len();
        let mut offset = 0usize;

        while offset < out_len {
            if step > 0
                && (step as i32) > 0
                && (phase as i32) >= 0
                && (sl as i32) > 0
                && ptr < pcm_len
            {
                let safe_samples = (sl as usize).min(pcm_len - ptr);
                let limit_phase = ((safe_samples as u64) << 12).min(0x8000_0000u64);
                if (phase as u64) < limit_phase {
                    let k_max = (limit_phase - 1 - (phase as u64)) / (step as u64);
                    let run = ((k_max + 1) as usize).min(out_len - offset);
                    mix_pcm_interior_run::<VOL64>(
                        &pcm_bytes[ptr..pcm_len],
                        &mut out[offset..offset + run],
                        phase,
                        step,
                        vol,
                    );
                    phase = phase.wrapping_add(step.wrapping_mul(run as u32));
                    offset += run;
                    continue;
                }
            }

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
            let scaled = if VOL64 {
                s >> 2
            } else {
                s.wrapping_mul(vol) >> 8
            };
            out[offset] = out[offset].wrapping_add(scaled as i8);
            offset += 1;
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
        if voice.volume == 0x40 {
            Self::mix_v2_impl::<true, true>(pcm_bytes, voice, out);
        } else {
            Self::mix_v2_impl::<true, false>(pcm_bytes, voice, out);
        }
    }

    /// Mix a **GBAModPlay v2.x word** ([`LogikVersion::V2Mod2Word`], *Zapper*) PCM8 voice
    /// into an 8-bit DirectSound mixing buffer.
    #[inline]
    pub fn mix_v2_word(pcm_bytes: &[u8], voice: &mut LogikPcmVoice, out: &mut [i8]) {
        if voice.volume == 0x40 {
            Self::mix_v2_impl::<false, true>(pcm_bytes, voice, out);
        } else {
            Self::mix_v2_impl::<false, false>(pcm_bytes, voice, out);
        }
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
        let out_len = out.len();
        let mut offset = 0usize;

        while offset < out_len {
            if step > 0
                && (step as i32) > 0
                && (phase as i32) >= 0
                && (sl as i32) > 0
                && ptr < pcm_len
            {
                let safe_pcm = ((pcm_len - ptr) as u64).saturating_mul(0x1000);
                let limit_phase = (sl as u64).min(safe_pcm);
                if (phase as u64) < limit_phase {
                    let k_max = (limit_phase - 1 - (phase as u64)) / (step as u64);
                    let run = ((k_max + 1) as usize).min(out_len - offset);
                    mix_pcm_interior_run::<true>(
                        &pcm_bytes[ptr..pcm_len],
                        &mut out[offset..offset + run],
                        phase,
                        step,
                        0x40,
                    );
                    phase = phase.wrapping_add(step.wrapping_mul(run as u32));
                    offset += run;
                    continue;
                }
            }

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
            out[offset] = out[offset].wrapping_add((s >> 2) as i8);
            offset += 1;
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
        let out_len = out.len();
        let mut offset = 0usize;

        while offset < out_len {
            if step > 0
                && (step as i32) > 0
                && (phase as i32) >= 0
                && (sl as i32) > 0
                && ptr < pcm_len
            {
                let safe_pcm = ((pcm_len - ptr) as u64).saturating_mul(0x1000);
                let limit_phase = (sl as u64).min(safe_pcm);
                if (phase as u64) < limit_phase {
                    let k_max = (limit_phase - 1 - (phase as u64)) / (step as u64);
                    let run = ((k_max + 1) as usize).min(out_len - offset);
                    mix_pcm_interior_run::<false>(
                        &pcm_bytes[ptr..pcm_len],
                        &mut out[offset..offset + run],
                        phase,
                        step,
                        vol,
                    );
                    phase = phase.wrapping_add(step.wrapping_mul(run as u32));
                    offset += run;
                    continue;
                }
            }

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
            out[offset] = out[offset].wrapping_add(scaled as i8);
            offset += 1;
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
        let mut byte_off = voice.byte_offset;
        let mut toggle = voice.nibble_toggle;
        let mut cached = voice.cached_byte;
        let mut rem = voice.remaining_nibbles;
        let mut out_idx = 0usize;
        let out_len = out.len();
        let data_len = adpcm_bytes.len();

        // If starting on the low nibble, consume 1 nibble first to align `toggle == 0`.
        if toggle != 0 && out_idx < out_len {
            let nib = cached & 0x0F;
            toggle = !toggle;

            pred = pred.wrapping_add(IMA_SIGNED_DIFF_LUT[step_idx as usize][nib as usize]);
            step_idx = (step_idx + ima_index_delta(nib)).clamp(0, 88);
            let scaled = ((pred.wrapping_mul(vol) as u32) >> 16) as i8;

            out[out_idx] = out[out_idx].wrapping_add(scaled);
            out_idx += 1;
            if out_idx < out_len {
                out[out_idx] = out[out_idx].wrapping_add(scaled);
                out_idx += 1;
                if out_idx < out_len {
                    rem = rem.wrapping_sub(1);
                    if rem == 0 {
                        voice.active = false;
                        voice.remaining_nibbles = rem;
                        voice.predictor = pred;
                        voice.step_index = step_idx;
                        voice.byte_offset = byte_off;
                        voice.nibble_toggle = toggle;
                        voice.cached_byte = cached;
                        return;
                    }
                }
            }
        }

        // Fast interior byte-pair path: decodes 2 nibbles -> 4 output samples per byte with
        // register-arithmetic step_idx updates (breaking L1 load-to-use latency) and
        // zero per-nibble toggle/bounds/termination branches.
        if toggle == 0 && out_idx + 4 < out_len && byte_off < data_len && rem > 2 {
            let avail_bytes = data_len - byte_off;
            let avail_out_bytes = (out_len - out_idx - 1) / 4;
            let avail_rem_bytes = ((rem - 1) / 2) as usize;
            let fast_bytes = avail_bytes.min(avail_out_bytes).min(avail_rem_bytes);

            if fast_bytes > 0 {
                let src = &adpcm_bytes[byte_off..byte_off + fast_bytes];
                let dst = &mut out[out_idx..out_idx + fast_bytes * 4];
                let mut last_byte = cached;
                let (dst_chunks, _) = dst.as_chunks_mut::<4>();
                for (&byte, chunk) in src.iter().zip(dst_chunks) {
                    last_byte = byte;

                    // High nibble -> samples 0 & 1
                    let hi = (byte >> 4) & 0x0F;
                    pred = pred.wrapping_add(IMA_SIGNED_DIFF_LUT[step_idx as usize][hi as usize]);
                    step_idx = (step_idx + ima_index_delta(hi)).clamp(0, 88);
                    let s_hi = ((pred.wrapping_mul(vol) as u32) >> 16) as i8;
                    chunk[0] = chunk[0].wrapping_add(s_hi);
                    chunk[1] = chunk[1].wrapping_add(s_hi);

                    // Low nibble -> samples 2 & 3
                    let lo = byte & 0x0F;
                    pred = pred.wrapping_add(IMA_SIGNED_DIFF_LUT[step_idx as usize][lo as usize]);
                    step_idx = (step_idx + ima_index_delta(lo)).clamp(0, 88);
                    let s_lo = ((pred.wrapping_mul(vol) as u32) >> 16) as i8;
                    chunk[2] = chunk[2].wrapping_add(s_lo);
                    chunk[3] = chunk[3].wrapping_add(s_lo);
                }
                cached = last_byte;
                byte_off += fast_bytes;
                out_idx += fast_bytes * 4;
                rem = rem.wrapping_sub((fast_bytes as u32) * 2);
            }
        }

        // Exact scalar tail (handles remaining 1..=4 samples, EOF, odd lengths, and rem == 0).
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

            pred = pred.wrapping_add(IMA_SIGNED_DIFF_LUT[step_idx as usize][nib as usize]);
            step_idx = (step_idx + ima_index_delta(nib)).clamp(0, 88);

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

        let gain = (volume.clamp(0.0, 1.0) as f64) * (1.0 / 128.0);
        let pan_clamped = pan.clamp(-1.0, 1.0) as f64;
        let angle = (pan_clamped + 1.0) * core::f64::consts::FRAC_PI_4;
        let gain_l = (gain * cos_f64(angle)) as f32;
        let gain_r = (gain * sin_f64(angle)) as f32;

        let loop_bounds = sample.loop_start.zip(sample.loop_end).and_then(|(ls, le)| {
            let ls = ls as usize;
            let le = (le as usize).min(pcm_len);
            if le > ls { Some((ls, le)) } else { None }
        });
        let (looped, loop_start, effective_end) = match loop_bounds {
            Some((ls, le)) => (true, ls, le),
            None => (false, 0, pcm_len),
        };
        let loop_len = if looped { effective_end - loop_start } else { 0 };
        let loop_start_f = loop_start as f64;
        let loop_len_f = loop_len as f64;
        let effective_end_f = effective_end as f64;

        let out_l = &mut out_left[..frames];
        let out_r = &mut out_right[..frames];

        if self.sinc_lobes == 3 {
            let safe_max_base = effective_end.saturating_sub(4);
            let mut ph = *pos_f64;
            let mut idx = 0usize;

            while idx < frames {
                if ph >= effective_end_f {
                    if looped {
                        ph -= loop_len_f;
                        if ph >= effective_end_f {
                            ph = loop_start_f + ((ph - loop_start_f) % loop_len_f);
                        }
                        if ph < loop_start_f {
                            ph = loop_start_f;
                        }
                    } else {
                        break;
                    }
                }

                let base = ph as usize;
                if effective_end >= 6
                    && base >= 2
                    && base <= safe_max_base
                    && step < (1u64 << 31) as f64
                {
                    let safe_limit = (safe_max_base as f64) + 0.999_999;
                    let max_interior = (((safe_limit - ph) / step) as usize) + 1;
                    let run = max_interior.min(frames - idx);
                    if run >= 4 {
                        const Q32_SCALE: f64 = 4_294_967_296.0;
                        let ph_q32 = (ph * Q32_SCALE) as u64;
                        let step_q32 = (step * Q32_SCALE) as u64;
                        let last_base = ((ph_q32
                            .saturating_add(step_q32.saturating_mul((run - 1) as u64)))
                            >> 32) as usize;
                        if last_base <= safe_max_base {
                            let src = &pcm[..last_base + 4];
                            let mut ph_shifted = ph_q32 - (2u64 << 32);
                            for (ol, or) in out_l[idx..idx + run]
                                .iter_mut()
                                .zip(&mut out_r[idx..idx + run])
                            {
                                let b0 = (ph_shifted >> 32) as usize;
                                // Nearest 1024-phase index in `0..=1024`
                                let p = (((ph_shifted & 0xFFFF_FFFF) + (1 << 21)) >> 22) as usize;
                                let w = &LANCZOS3_LUT[p];
                                let win = &src[b0..b0 + 6];
                                let s0 = (win[0] as i8) as f32;
                                let s1 = (win[1] as i8) as f32;
                                let s2 = (win[2] as i8) as f32;
                                let s3 = (win[3] as i8) as f32;
                                let s4 = (win[4] as i8) as f32;
                                let s5 = (win[5] as i8) as f32;
                                // Balanced reduction tree halves FMA dependency chain latency
                                let val = (s0 * w[0] + s1 * w[1])
                                    + (s2 * w[2] + s3 * w[3])
                                    + (s4 * w[4] + s5 * w[5]);
                                *ol += val * gain_l;
                                *or += val * gain_r;
                                ph_shifted += step_q32;
                            }
                            ph += (run as f64) * step;
                            idx += run;
                            continue;
                        }
                    }
                }

                // Boundary / loop-edge single-sample step
                let frac = ph - (base as f64);
                let p = ((frac * (LANCZOS_PHASES as f64) + 0.5) as usize).min(LANCZOS_PHASES);
                let w = &LANCZOS3_LUT[p];

                let mut s = [0.0f32; 6];
                let base_i = base as isize;
                for (slot, k) in s.iter_mut().zip(-2isize..=3) {
                    let sample_idx = base_i + k;
                    if sample_idx < 0 {
                        if looped {
                            *slot = (pcm[0] as i8) as f32;
                        }
                    } else {
                        let u = sample_idx as usize;
                        if u < effective_end {
                            *slot = (pcm[u] as i8) as f32;
                        } else if looped {
                            let wrapped = loop_start + ((u - loop_start) % loop_len);
                            *slot = (pcm[wrapped] as i8) as f32;
                        }
                    }
                }

                let val = (s[0] * w[0] + s[1] * w[1])
                    + (s[2] * w[2] + s[3] * w[3])
                    + (s[4] * w[4] + s[5] * w[5]);

                out_l[idx] += val * gain_l;
                out_r[idx] += val * gain_r;
                ph += step;
                idx += 1;
            }

            *pos_f64 = ph;
            return;
        }

        // General fallback for custom `sinc_lobes != 3`
        let a = self.sinc_lobes.max(1) as isize;
        let a_f = a as f64;
        let mut ph = *pos_f64;

        for (ol, or) in out_l.iter_mut().zip(out_r.iter_mut()) {
            if ph >= effective_end_f {
                if looped {
                    ph = loop_start_f + ((ph - loop_start_f) % loop_len_f);
                } else {
                    break;
                }
            }
            let base = ph as isize;
            let frac = ph - (base as f64);
            let mut acc = 0.0f64;
            let mut wsum = 0.0f64;
            let mut k = 1 - a;
            while k <= a {
                let sample_idx = base + k;
                let s_val = if sample_idx < 0 {
                    if looped {
                        (pcm[0] as i8) as f64
                    } else {
                        0.0
                    }
                } else {
                    let u = sample_idx as usize;
                    if u < effective_end {
                        (pcm[u] as i8) as f64
                    } else if looped {
                        let wrapped = loop_start + ((u - loop_start) % loop_len);
                        (pcm[wrapped] as i8) as f64
                    } else {
                        0.0
                    }
                };
                let w = lanczos_kernel(frac - (k as f64), a_f);
                acc += w * s_val;
                wsum += w;
                k += 1;
            }
            let interpolated = if wsum > 1e-12 {
                (acc / wsum) as f32
            } else {
                0.0
            };

            *ol += interpolated * gain_l;
            *or += interpolated * gain_r;
            ph += step;
        }

        *pos_f64 = ph;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::freq::IMA_INDEX_TABLE;

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

    #[test]
    fn test_ima_signed_diff_lut_matches_arm_formula() {
        for idx in 0..89usize {
            let step = IMA_STEP_TABLE[idx];
            for nib in 0..16usize {
                let expected_delta = {
                    let diff = ((((nib & 7) as i32) * step) as u32 >> 2) as i32;
                    if (nib & 8) != 0 { -diff } else { diff }
                };
                assert_eq!(IMA_SIGNED_DIFF_LUT[idx][nib], expected_delta);
                assert_eq!(ima_index_delta(nib as u8), IMA_INDEX_TABLE[nib]);
            }
        }
    }

    #[test]
    fn test_studio_synth_lut_matches_direct_lanczos3() {
        let pcm: [u8; 64] = core::array::from_fn(|i| {
            let angle = (i as f64) * 0.35;
            (sin_const(angle) * 100.0) as i8 as u8
        });
        let view = LogikSampleView::from_pcm_bytes(10_512, 64, None, None, &pcm);
        let synth = StudioSynth::new(48_000);
        let mut left = [0.0f32; 128];
        let mut right = [0.0f32; 128];
        let mut pos = 5.25;
        synth.render_voice_f32(&view, &mut pos, 12_000.0, 1.0, 0.0, &mut left, &mut right);

        let step = 12_000.0 / 48_000.0;
        let gain_l = ((1.0 / 128.0) * cos_f64(core::f64::consts::FRAC_PI_4)) as f32;
        let mut ref_pos = 5.25f64;
        for &actual in &left {
            let center = ref_pos as isize;
            let mut acc = 0.0f64;
            let mut wsum = 0.0f64;
            for k in (center - 2)..=(center + 3) {
                let s = (pcm[k as usize] as i8) as f64;
                let w = lanczos_kernel(ref_pos - (k as f64), 3.0);
                acc += s * w;
                wsum += w;
            }
            let expected = ((acc / wsum) as f32) * gain_l;
            assert!(
                (actual - expected).abs() < 1e-3,
                "LUT mismatch: actual={actual}, expected={expected}"
            );
            ref_pos += step;
        }
    }
}
