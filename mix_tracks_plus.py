#!/usr/bin/env python3

"""

Auto DJ Mix ++ (subset selection, multiple variants, smart entry points)



Features

- Beat-align + bar-length equal-power crossfades (4/4 assumed)

- Harmonic key detection + optional harmonic ordering

- Smart "mix-in" moments: finds strong onset peaks (drops/chorus) and aligns new track so the peak lands just after your crossfade

- Target runtime: chooses a subset of tracks that best fits --target_minutes (knapsack-like heuristic)

- Multiple variants: generate N different mixes by rotating start choices & small neighborhood swaps while respecting harmonic distance

- Optional Rubber Band time-stretch via pyrubberband for better quality



Usage
  python mix_tracks_plus.py --input ./input --outdir ./out --target_bpm 172 \
    --bars 16 --target_minutes 50 --variants 4 --harmonic_order --order_start auto \
    --cue_csv --use_rubberband --debug_clicks


Outputs

  out/mix_v001.wav, out/mix_v002.wav, ... plus cue CSVs per variant

"""



import argparse, os, glob, math, random, csv, shutil, subprocess, tempfile

import numpy as np

import soundfile as sf

import librosa

import librosa.effects

from dataclasses import dataclass

from typing import List, Tuple



try:
    import pyrubberband as pyrb
    HAS_PYRB_MODULE = True
except Exception:
    pyrb = None
    HAS_PYRB_MODULE = False

_ENV_RB_PATH = os.environ.get("RUBBERBAND_PATH", "").strip()
if _ENV_RB_PATH and os.path.exists(_ENV_RB_PATH):
    RUBBERBAND_BIN = _ENV_RB_PATH
else:
    RUBBERBAND_BIN = shutil.which("rubberband")

HAS_RUBBERBAND = bool(RUBBERBAND_BIN)
WARNED_NO_RUBBERBAND = False

def warn_once_no_rubberband():
    global WARNED_NO_RUBBERBAND
    if WARNED_NO_RUBBERBAND:
        return
    print("[warn] --use_rubberband requested but no Rubber Band CLI was found on PATH; using librosa time-stretch instead.")
    WARNED_NO_RUBBERBAND = True


random.seed(42)



# at top (once)

ANALYSIS_NFFT = 1024

ANALYSIS_HOP  = 512



# ---- replace peak_pick with a simple, robust peak finder ----

def simple_peaks(x, distance=16, threshold=None):

    # distance ~ how many frames between peaks (16 ~= ~8 beats at hop=512, sr~44.1k)

    if threshold is None:

        threshold = float(np.mean(x) + 0.5*np.std(x))

    peaks = []

    last = -distance

    for i in range(1, len(x)-1):

        if x[i] > x[i-1] and x[i] > x[i+1] and x[i] >= threshold and (i - last) >= distance:

            peaks.append(i)

            last = i

    return np.array(peaks, dtype=int)



PITCHES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

_KK_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)

_KK_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)



@dataclass
class TrackInfo:
    path: str
    name: str
    y: np.ndarray
    sr: int
    duration: float
    bpm: float
    beats: np.ndarray
    beat_times: np.ndarray
    first_downbeat_time: float
    key_pc: int
    key_mode: str
    key_name: str
    peak_times: np.ndarray  # candidate musical peaks (seconds)
    onset_env: np.ndarray   # onset envelope (for stability analysis)
    onset_times: np.ndarray # times for onset_env frames
    anchor_time: float      # track's main beat anchor (seconds, pre-stretch)

def kk_detect_key(y: np.ndarray, sr: int) -> Tuple[int, str, str]:

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=12)

    prof = chroma.mean(axis=1); prof = prof/(prof.sum()+1e-12)

    best = (-1e9, 0, 'major')

    for mode, tmpl in [('major','major'), ('minor','minor')]:

        T = _KK_MAJOR if mode=='major' else _KK_MINOR

        T = T / T.sum()

        for pc in range(12):

            corr = float(np.dot(prof, np.roll(T, pc)))

            if corr > best[0]:

                best = (corr, pc, mode)

    _, pc, mode = best

    return pc, mode, f"{PITCHES[pc]} {'major' if mode=='major' else 'minor'}"

def analyze_track(path: str) -> TrackInfo:

    # 1) Read stereo at native SR, float32

    y_stereo, sr = sf.read(path, always_2d=True)

    y_stereo = y_stereo.astype(np.float32, copy=False)

    duration = len(y_stereo) / sr



    # 2) Make a MONO copy for all librosa analysis (prevents giant arrays)

    y_mono = librosa.to_mono(y_stereo.T).astype(np.float32, copy=False)

    if len(y_mono) < 4096:

        # skip pathological/very short files

        raise RuntimeError(f"Audio too short for analysis: {os.path.basename(path)}")



    # 3) Beat tracking / onset with smaller FFT to avoid n_fft warnings

    tempo, beats = librosa.beat.beat_track(y=y_mono, sr=sr, trim=False)

    tempo_val = float(np.atleast_1d(tempo)[0])

    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=ANALYSIS_HOP)



    onset_env = librosa.onset.onset_strength(

        y=y_mono, sr=sr, hop_length=ANALYSIS_HOP, n_fft=ANALYSIS_NFFT

    )

    onset_peaks = simple_peaks(onset_env, distance=16)

    times_env = librosa.frames_to_time(

        np.arange(len(onset_env)), sr=sr, hop_length=ANALYSIS_HOP

    )
    peak_times = times_env[onset_peaks] if len(onset_peaks) else np.array([duration*0.25], dtype=np.float32)



    # first downbeat-ish guess

    beat_frames = beats[:min(len(beats), 8)]

    if len(beat_frames) > 0:

        strengths = onset_env[beat_frames]

        idx = int(np.argmax(strengths)); first_downbeat_time = float(beat_times[idx])

    else:

        first_downbeat_time = 0.0



    key_pc, key_mode, key_name = kk_detect_key(y_mono, sr)

    return TrackInfo(
        path=path,
        name=os.path.basename(path),
        y=y_stereo, sr=sr, duration=duration,
        bpm=tempo_val,
        beats=beats, beat_times=beat_times,
        first_downbeat_time=first_downbeat_time,
        key_pc=key_pc, key_mode=key_mode, key_name=key_name,
        peak_times=peak_times,
        onset_env=onset_env.astype(np.float32, copy=False),
        onset_times=times_env.astype(np.float32, copy=False),
        anchor_time=first_downbeat_time,  # temp; replaced later per --anchor
    )

def _nearest_beat_time(beat_times: np.ndarray, t: float) -> float:
    if beat_times is None or len(beat_times) == 0:
        return float(max(0.0, t))
    i = int(np.argmin(np.abs(beat_times - t)))
    return float(beat_times[i])

def find_stable_groove(tr: TrackInfo, target_bpm: float,
                       allow_multiples: bool = True,
                       bpm_tol: float = 0.07,
                       smooth_frames: int = 9,
                       thr_rel: float = 0.55,
                       min_bars: int = 8,
                       skip_bars: int = 16) -> Tuple[float, Tuple[float,float]]:
    """
    Return (anchor_time, (seg_start, seg_end)) for the first stable-beat segment
    after skip_bars, where stability >= thr_rel of max and length >= min_bars.
    If none found, return (np.nan, (np.nan, np.nan)).
    """
    sr = tr.sr
    hop = ANALYSIS_HOP
    env = tr.onset_env
    times = tr.onset_times
    if env is None or len(env) < 32:
        return (np.nan, (np.nan, np.nan))

    # tempogram (autocorrelation) in tempo domain
    Tg = librosa.feature.tempogram(onset_envelope=env, sr=sr, hop_length=hop)
    # corresponding BPMs
    ac_times = librosa.times_like(env, sr=sr, hop_length=hop)  # same length as times
    bpms = librosa.tempo_frequencies(Tg.shape[0], sr=sr, hop_length=hop)

    # choose bpm bins near target (optionally include 0.5x and 2x)
    targets = [target_bpm]
    if allow_multiples:
        targets += [target_bpm*0.5, target_bpm*2.0]
    mask = np.zeros_like(bpms, dtype=bool)
    for tb in targets:
        mask |= (np.abs(bpms - tb) <= (bpm_tol * tb))
    if not np.any(mask):
        # widen if nothing is selected
        mask = (bpms >= 0.5*target_bpm) & (bpms <= 2.0*target_bpm)

    # stability: max tempogram energy within selected bpm bins at each time
    stab = Tg[mask, :].max(axis=0)
    # smooth
    if smooth_frames > 1:
        k = smooth_frames
        pad = k//2
        stab = np.pad(stab, (pad,pad), mode="edge")
        stab = np.convolve(stab, np.ones(k)/k, mode="valid")

    # threshold
    if len(stab) == 0: return (np.nan, (np.nan, np.nan))
    thr = float(thr_rel * np.max(stab))
    good = stab >= thr

    # convert bars to seconds
    bar_sec = 60.0/target_bpm*4.0
    min_len_sec = max(1.0, min_bars * bar_sec)
    skip_sec = max(0.0, skip_bars * bar_sec)

    # scan contiguous segments above threshold
    segs = []
    i = 0
    while i < len(good):
        if not good[i]:
            i += 1
            continue
        j = i + 1
        while j < len(good) and good[j]:
            j += 1
        t0, t1 = times[i], times[min(j, len(times)-1)]
        if t1 - t0 >= min_len_sec and t1 > skip_sec:
            segs.append((t0, t1))
        i = j
    if not segs:
        return (np.nan, (np.nan, np.nan))

    # pick earliest segment that starts after skip_sec (or overlaps it)
    segs.sort(key=lambda x: x[0])
    seg = segs[0]
    # anchor near the beginning of that segment so crossfades land in tight groove
    anchor_guess = max(seg[0], skip_sec)
    anchor = _nearest_beat_time(tr.beat_times, anchor_guess)
    return (float(anchor), (float(seg[0]), float(seg[1])))

def choose_anchor_time(tr: TrackInfo, target_bpm: float, mode: str = "auto", skip_bars: int = 16,
                       stable_min_bars: int = 8) -> float:
    bar_sec = 60.0/target_bpm*4.0
    if mode == "intro":
        return float(tr.first_downbeat_time)
    # ignore very early bars if we want a later anchor
    min_t = skip_bars * bar_sec
    # STABLE mode (implicit in AUTO): prefer first stable groove if available
    if mode in ("auto",):
        anchor, seg = find_stable_groove(
            tr, target_bpm, allow_multiples=True, bpm_tol=0.07,
            smooth_frames=9,
            thr_rel=0.55, min_bars=stable_min_bars, skip_bars=skip_bars
        )
        if not np.isnan(anchor):
            return anchor
    # PEAK mode: use the first strong musical peak after skip
    if mode == "peak":
        cands = [p for p in tr.peak_times if p >= min_t]
        t = (cands[0] if len(cands) else max(tr.first_downbeat_time, min_t))
        return _nearest_beat_time(tr.beat_times, t)
    # MID mode: anchor near the middle (often stable drums)
    if mode == "mid":
        t = max(min_t, 0.5 * tr.duration)
        return _nearest_beat_time(tr.beat_times, t)
    # AUTO fallback: first strong peak after skip, else mid
    cands = [p for p in tr.peak_times if p >= min_t]
    if len(cands):
        return _nearest_beat_time(tr.beat_times, cands[0])
    return _nearest_beat_time(tr.beat_times, max(min_t, 0.5 * tr.duration))

def time_stretch_multi(y: np.ndarray, sr: int, rate: float, use_rubberband: bool) -> np.ndarray:

    if abs(rate - 1.0) < 1e-3 or y.shape[0] < 4096:

        return y



    def _align_and_stack(channels):

        min_len = min(len(ch) for ch in channels)

        channels = [ch[:min_len] for ch in channels]

        return np.stack(channels, axis=1)



    def _librosa_stretch():

        chans = [librosa.effects.time_stretch(y[:, c], rate=rate) for c in range(y.shape[1])]

        return _align_and_stack(chans)



    def _pyrb_stretch():

        chans = [pyrb.time_stretch(y[:, c], sr, rate) for c in range(y.shape[1])]

        return _align_and_stack(chans)



    def _rubberband_cli_stretch():

        with tempfile.TemporaryDirectory(prefix="rb_cli_") as tmp:

            inp = os.path.join(tmp, "in.wav")

            out = os.path.join(tmp, "out.wav")

            sf.write(inp, y, sr)

            cmd = [RUBBERBAND_BIN, "--tempo", f"{rate:.8f}", inp, out]

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stretched, sr_out = sf.read(out, always_2d=True)

            if sr_out != sr:

                stretched = hq_resample(stretched, sr_out, sr)

            return stretched.astype(np.float32, copy=False)



    if use_rubberband:
        if RUBBERBAND_BIN:
            try:
                return _rubberband_cli_stretch()
            except Exception as e:
                print(f"[warn] rubberband CLI failed ({e}); trying pyrubberband/librosa")
            if HAS_PYRB_MODULE:
                try:
                    return _pyrb_stretch()
                except Exception as e:
                    print(f"[warn] pyrubberband failed ({e}); falling back to librosa")
        else:
            warn_once_no_rubberband()

    return _librosa_stretch()


def hq_resample(y: np.ndarray, sr_src: int, sr_dst: int) -> np.ndarray:

    if sr_src == sr_dst: return y

    # y shape: (N, C)

    chans = []

    for c in range(y.shape[1]):

        chans.append(librosa.resample(y[:, c], orig_sr=sr_src, target_sr=sr_dst, res_type="kaiser_best"))

    y2 = np.stack(chans, axis=1)

    return y2



def fold_to_target(src_bpm: float, tgt_bpm: float) -> float:
    # choose src, 2x, or 0.5x - whichever is closest to target
    if src_bpm <= 0 or tgt_bpm <= 0: return src_bpm

    cands = [src_bpm, src_bpm*2.0, src_bpm*0.5]

    return min(cands, key=lambda v: abs(v - tgt_bpm))



def compute_rate(src_bpm: float, tgt_bpm: float, tol: float = 0.02) -> float:

    """Return 1.0 if already close to target (or its x2 / half-speed), else the stretch rate."""

    eff = fold_to_target(src_bpm, tgt_bpm)

    if eff <= 0: return 1.0

    rate = tgt_bpm / eff

    # if within +/-2% already, don't stretch
    if abs(rate - 1.0) <= tol: return 1.0

    return rate



def normalize_peak(y: np.ndarray, peak_db=-1.0) -> np.ndarray:

    peak = np.max(np.abs(y)) + 1e-12

    target = 10**(peak_db/20.0)

    return y * (target/peak)



def equal_power_xfade(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    n = min(len(a), len(b))

    a = a[:n]

    b = b[:n]

    # time base

    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=a.dtype)

    if a.ndim == 1:            # mono

        ga = np.cos(t * np.pi/2)

        gb = np.cos((1.0 - t) * np.pi/2)

    else:                      # stereo/multichannel: (N, C)

        ga = np.cos(t * np.pi/2)[:, None]

        gb = np.cos((1.0 - t) * np.pi/2)[:, None]

    return a * ga + b * gb



# Harmonic helpers

def fifth_distance(pc_a:int, pc_b:int) -> int:

    order = [0]; cur = 0

    for _ in range(11):

        cur = (cur + 7) % 12; order.append(cur)

    pos = {p:i for i,p in enumerate(order)}

    diff = abs(pos[pc_a] - pos[pc_b])

    return min(diff, 12-diff)



def harmonic_distance(a: TrackInfo, b: TrackInfo) -> float:

    if a.key_pc == b.key_pc and a.key_mode == b.key_mode: return 0.0

    if a.key_pc == b.key_pc and a.key_mode != b.key_mode: return 0.5

    rel = ((a.key_mode=='major' and b.key_mode=='minor') or (a.key_mode=='minor' and b.key_mode=='major'))

    if rel and ((b.key_pc - a.key_pc) % 12 in (3,9) or (a.key_pc - b.key_pc) % 12 in (3,9)): return 0.75

    if a.key_mode == b.key_mode and ((b.key_pc - a.key_pc) % 12 in (7,5) or (a.key_pc - b.key_pc) % 12 in (7,5)): return 1.0

    d = fifth_distance(a.key_pc, b.key_pc)

    if a.key_mode != b.key_mode: d += 0.25

    return float(d)



def quantize_time(t_sec: float, bpm: float, bars: int = 1) -> float:

    # bars=1 => snap to bar; use bars=0.25 to snap to beats if you prefer

    bar_len = (60.0 / bpm) * 4.0

    q = bars * bar_len

    return round(t_sec / q) * q



def bar_len(bpm: float) -> float:

    return (60.0 / bpm) * 4.0



def beat_len(bpm: float) -> float:

    return 60.0 / bpm



def prev_global_grid_start(mix_len_smp: int, sr: int, bpm: float, grid: str = "bar") -> int:
    """Return the previous boundary (in samples) on the global mix grid (bar or beat)."""
    unit = (bar_len(bpm) if grid == "bar" else beat_len(bpm)) * sr
    return int(math.floor(mix_len_smp / unit) * unit)


def micro_align(tailA: np.ndarray, headB: np.ndarray, sr: int, bpm: float, max_frac: float = 0.5) -> int:
    """
    Find a tiny head shift (<= 1/8 beat) that best aligns onsets.
    Works on mono reductions; uses onset_strength envelopes and plain dot-product scoring.
    Returns a shift in samples to apply to the INCOMING head (positive = shift head later).
    """
    # allow up to ±(max_frac) of a beat
    max_shift = int(max_frac * (60.0 / bpm) * sr)
    if max_shift < 32:
        return 0

    # focus on percussive component for transient alignment
    Amono = tailA.mean(axis=1) if tailA.ndim == 2 else tailA
    Bmono = headB.mean(axis=1) if headB.ndim == 2 else headB
    try:
        Ap, _ = librosa.effects.hpss(Amono.astype(np.float32, copy=False))
        Bp, _ = librosa.effects.hpss(Bmono.astype(np.float32, copy=False))
    except Exception:
        Ap, Bp = Amono, Bmono

    # build short onset envelopes (center=False so indices stay aligned)
    hop = 256  # finer resolution
    nfft = 1024
    Ae = librosa.onset.onset_strength(y=Ap.astype(np.float32, copy=False), sr=sr,
                                      hop_length=hop, n_fft=nfft, center=False)
    Be = librosa.onset.onset_strength(y=Bp.astype(np.float32, copy=False), sr=sr,
                                      hop_length=hop, n_fft=nfft, center=False)
    if len(Ae) < 8 or len(Be) < 8:
        return 0



    # normalize envelopes

    Ae = (Ae - Ae.mean()) / (Ae.std() + 1e-9)

    Be = (Be - Be.mean()) / (Be.std() + 1e-9)



    # convert sample shifts to envelope-frame shifts

    smp_per_frame = hop

    max_fshift = max(1, max_shift // smp_per_frame)



    # compare small shifts [-max_fshift, +max_fshift]

    best_corr, best_shift_frames = -1e9, 0

    L = min(len(Ae), len(Be))

    Ae = Ae[-L:]

    Be = Be[:L]



    for s in range(-max_fshift, max_fshift + 1):

        if s == 0:

            Av = Ae[-L:]

            Bv = Be[:L]

        elif s > 0:

            Bv = Be[s:s+L]

            Av = Ae[-len(Bv):]

        else:  # s < 0

            Av = Ae[-(L + s):]

            Bv = Be[:len(Av)]

        if len(Av) < 8 or len(Bv) < 8:

            continue

        L2 = min(len(Av), len(Bv))

        Av = Av[-L2:]; Bv = Bv[:L2]

        c = float(np.dot(Av, Bv))

        if c > best_corr:

            best_corr = c

            best_shift_frames = s

    # convert back to SAMPLES
    return int(best_shift_frames * smp_per_frame)


def snap_to_track_grid(t_sec: float, tr: TrackInfo, target_bpm: float, grid: str = "bar") -> float:
    """Snap a time (seconds, pre-stretch) to the track's own grid (bar or beat),
    using folded BPM so grid lines map 1:1 after stretching to target_bpm."""
    eff_bpm = fold_to_target(tr.bpm, target_bpm)
    if eff_bpm <= 0:
        return t_sec
    unit = bar_len(eff_bpm) if grid == "bar" else beat_len(eff_bpm)
    # anchor is the first strong downbeat snapped to the track's bar grid
    # (use detected downbeat; if absent, just use 0)
    anchor = tr.first_downbeat_time
    # snap anchor itself to grid to stabilize phase
    anchor = round(anchor / unit) * unit
    # now snap t_sec relative to that anchor
    return anchor + round((t_sec - anchor) / unit) * unit


def add_click(mono_len, sr, bpm, first_click_sec=0.0, gain=0.15):
    """Optional debug click track to verify phase alignment. Mute/remove after testing."""

    bl = bar_len(bpm)

    beat = bl / 4.0

    y = np.zeros(mono_len, dtype=np.float32)

    t0 = int(first_click_sec * sr)

    i = t0

    while i < mono_len:

        y[i:i+int(0.002*sr)] += 1.0  # 2ms tick

        i += int(beat * sr)

    y = np.clip(y, -1, 1)

    return y * gain



def harmonic_order(tracks: List[TrackInfo], start: str = "auto") -> List[int]:

    n = len(tracks)

    if n <= 2: return list(range(n))

    D = np.zeros((n,n), dtype=float)

    for i in range(n):

        for j in range(n):

            D[i,j] = 0.0 if i==j else harmonic_distance(tracks[i], tracks[j])

    cur = int(np.argmin(D.sum(axis=1))) if start=="auto" else 0

    order, used = [cur], {cur}

    while len(order)<n:

        j = min((j for j in range(n) if j not in used), key=lambda k: D[cur,k])

        order.append(j); used.add(j); cur = j

    return order


# Entry-point helper: choose a "drop/chorus" peak and align so it lands just after the crossfade
def choose_entry_offset(tr, target_bpm, bars, min_bars=8, max_bars=128, ahead_bars=2, min_tail_bars=8, grid="bar") -> float:
    bar_sec   = 60.0/target_bpm*4.0
    xfade_sec = bars*bar_sec
    lower, upper = min_bars*bar_sec, max_bars*bar_sec
    candidates = [p for p in tr.peak_times if p>lower and p<tr.duration-2.0]
    peak = candidates[0] if candidates else max(tr.first_downbeat_time, bar_sec)
    desired_start = peak - xfade_sec - ahead_bars*bar_sec
    min_remaining = xfade_sec + min_tail_bars*bar_sec
    latest_start  = max(0.0, tr.duration - min_remaining)
    start_sec     = max(0.0, min(desired_start, latest_start))
    # snap to THIS TRACK's own bar grid (pre-stretch), not the global target grid
    return snap_to_track_grid(start_sec, tr, target_bpm, grid=grid)


# Subset selection to fit target minutes (greedy with look-ahead swaps)
def select_subset(tracks: List[TrackInfo], target_minutes: float, bars:int, target_bpm:float) -> List[int]:
    if not target_minutes or target_minutes <= 0: 
        return list(range(len(tracks)))
    total_target = target_minutes*60.0
    bar_sec = 60.0/target_bpm*4.0
    xfade_sec = bars*bar_sec
    slack = 60.0

    # Simple weight = duration - xfade (effective contribution)
    items = [(i, max(10.0, tr.duration - xfade_sec)) for i,tr in enumerate(tracks)]
    # Sort by (shorter first) to pack more variety, lightly weighted by harmonic centrality
    Dsum = [0.0]*len(tracks)
    for i in range(len(tracks)):
        for j in range(len(tracks)):
            if i!=j: Dsum[i]+=harmonic_distance(tracks[i], tracks[j])
    items.sort(key=lambda x: (x[1] + 0.05*Dsum[x[0]]))

    picked, time_sum = [], 0.0
    for idx, eff in items:
        if time_sum + eff <= total_target + slack:
            picked.append(idx)
            time_sum += eff

    if time_sum < total_target - slack:
        for idx, eff in items:
            if idx in picked:
                continue
            if time_sum + eff <= total_target + slack:
                picked.append(idx)
                time_sum += eff
            if time_sum >= total_target - slack/2:
                break

    if not picked:
        picked = [items[0][0]]
    return picked


def stitch(tracks, sequence, target_bpm, bars, use_rubberband,
           entry_min_bars=8, entry_max_bars=128, entry_ahead_bars=2,
           min_tail_bars=8, debug_clicks=False, debug_click_gain=0.12, grid="bar",
           strict_bpm=False):
    assert sequence, "Empty sequence"
    mix_sr = tracks[sequence[0]].sr
    bar_sec = 60.0/target_bpm*4.0
    bar_smp = max(1, int(round(bar_sec * mix_sr)))
    xfade = int(bars * bar_sec * mix_sr)
    tail_guard = int(min_tail_bars * bar_sec * mix_sr)
    safe_segment = xfade + tail_guard

    # --- first track ---
    first = tracks[sequence[0]]
    y_prev = hq_resample(first.y, first.sr, mix_sr)
    rate0  = compute_rate(first.bpm, target_bpm, tol=(0.0 if strict_bpm else 0.02))
    y_prev = time_stretch_multi(y_prev, mix_sr, rate0, use_rubberband)

    # anchor first track at bar boundary 1 bar before strong downbeat, on its own grid (pre-stretch)
    start_off_sec = max(0.0, first.first_downbeat_time - bar_sec)
    start_off_sec = snap_to_track_grid(start_off_sec, first, target_bpm, grid=grid)
    start_off_smp = int((start_off_sec / max(rate0, 1e-9)) * mix_sr)
    max_head_trim = max(0, len(y_prev) - safe_segment)
    start_off_smp = max(0, min(start_off_smp, max_head_trim))
    y_prev = y_prev[start_off_smp:]
    if len(y_prev) < safe_segment:
        print(f"[warn] {first.name} shorter than crossfade safety window; expect a quicker swap.")
    mix = y_prev.copy()
    cues = [(first.name, 0.0)]
    current_track_start = 0

    # --- following tracks ---
    for k in range(1, len(sequence)):
        tr = tracks[sequence[k]]
        y = hq_resample(tr.y, tr.sr, mix_sr)
        rate = compute_rate(tr.bpm, target_bpm, tol=0.02)
        y = time_stretch_multi(y, mix_sr, rate, use_rubberband)

        # enforce outgoing tail safety (A tail needs xfade + guard)
        min_start_allowed = current_track_start + safe_segment
        window_end = max(0, len(mix) - xfade)
        xfade_start_glob = prev_global_grid_start(window_end, mix_sr, target_bpm, grid=grid)
        if xfade_start_glob < min_start_allowed:
            bars_forward = math.ceil(max(0, min_start_allowed - xfade_start_glob) / bar_smp)
            candidate = xfade_start_glob + bars_forward * bar_smp
            if candidate <= window_end:
                xfade_start_glob = candidate
            else:
                print(f"[warn] Transition {k} forced earlier than safety window (track too short).")
                xfade_start_glob = window_end
        xfade_start_glob = prev_global_grid_start(xfade_start_glob, mix_sr, target_bpm, grid=grid)
        available_A = len(mix) - xfade_start_glob
        if available_A < xfade:
            print(f"[warn] Transition {k} using shorter tail ({available_A/mix_sr:.2f}s).")

        # pick an entry in the incoming track (pre-stretch seconds) snapped to its own bar grid
        off_sec_pre = choose_entry_offset(tr, target_bpm, bars,
                                          entry_min_bars, entry_max_bars,
                                          entry_ahead_bars, min_tail_bars=min_tail_bars, grid=grid)
        anchor_samples = int((off_sec_pre / max(rate, 1e-9)) * mix_sr)
        max_head_keep = max(0, len(y) - safe_segment)
        off_samples = max(0, min(anchor_samples, max_head_keep))
        if len(y) - off_samples < safe_segment:
            print(f"[warn] {tr.name} entry trimmed with only {(len(y) - off_samples)/mix_sr:.2f}s left.")
        y = y[off_samples:]

        xf = min(xfade, len(y), available_A)
        bar_aligned = (xf // bar_smp) * bar_smp
        if bar_aligned >= bar_smp:
            xf = bar_aligned
        if xf <= 0:
            print(f"[warn] Transition {k} skipped (no overlap).")
            continue

        tailA = mix[xfade_start_glob:xfade_start_glob + xf]
        preA  = mix[:xfade_start_glob]

        headB = y[:xf]

        # tiny sub-beat shift of the incoming head (positive = delay B)
        shift = micro_align(tailA, headB, mix_sr, target_bpm, max_frac=0.5)
        if shift > 0:
            if shift < len(y):
                y = y[shift:]
            else:
                pad = np.zeros((shift - len(y) + xf, y.shape[1] if y.ndim==2 else 1), dtype=y.dtype)
                y = np.concatenate([y, pad], axis=0)
                y = y[shift:]
            if len(y) < xf:
                pad = np.zeros((xf - len(y), y.shape[1]), dtype=y.dtype)
                y = np.concatenate([y, pad], axis=0)
            headB = y[:xf]
        elif shift < 0:
            pad = np.zeros((-shift, y.shape[1] if y.ndim==2 else 1), dtype=y.dtype)
            y   = np.concatenate([pad, y], axis=0)
            headB = y[:xf]

        xmix = equal_power_xfade(tailA, headB)
        restB = y[xf:]
        cue_start = len(preA)/mix_sr
        cues.append((tr.name, cue_start))
        mix = np.concatenate([preA, xmix, restB], axis=0)
        current_track_start = len(preA)

    if debug_clicks:
        click = add_click(len(mix), mix_sr, target_bpm, gain=float(debug_click_gain))
        if mix.ndim == 1:
            mix = np.clip(mix + click, -1.0, 1.0)
        else:
            click2 = click[:, None]
            mix = np.clip(mix + click2, -1.0, 1.0)

    mix = normalize_peak(mix, peak_db=-1.0)
    return mix, mix_sr, cues


def enforce_target_length(mix: np.ndarray, sr:int, target_minutes: float, outro_bars:int, target_bpm:float) -> np.ndarray:
    if not target_minutes or target_minutes<=0:
        return mix
    target_sec = target_minutes*60.0
    upper_sec = target_sec + 60.0  # +1 minute slack
    length_sec = len(mix)/sr
    if length_sec <= upper_sec:
        return mix

    trim_samples = int(min(upper_sec * sr, len(mix)))
    trim_samples = prev_global_grid_start(trim_samples, sr, target_bpm)
    trim_samples = max(trim_samples, 1)
    mix = mix[:trim_samples]

    bar_sec = 60.0/target_bpm*4.0
    fade = int(max(1, outro_bars) * bar_sec * sr)
    fade = min(fade, len(mix))
    if fade > 0:
        t = np.linspace(0,1,fade,endpoint=False)
        window = np.cos(t*np.pi/2.0)
        if mix.ndim == 1:
            mix[-fade:] *= window
        else:
            mix[-fade:] *= window[:, None]
    return mix


def make_variants(all_tracks: List[TrackInfo], base_order: List[int], subset_idx: List[int], variants:int) -> List[List[int]]:

    # Keep relative harmonic flow but rotate start and swap neighbors a bit for variety

    base = [i for i in base_order if i in subset_idx]

    if not base: base = subset_idx[:]

    outs = []

    for v in range(variants):

        rot = (v*max(1,len(base)//variants)) % len(base)

        seq = base[rot:] + base[:rot]

        # local swaps

        for _ in range(min(3, len(seq)//4)):

            a = random.randint(0, len(seq)-2)

            if harmonic_distance(all_tracks[seq[a]], all_tracks[seq[a+1]]) <= 1.25:

                seq[a], seq[a+1] = seq[a+1], seq[a]

        outs.append(seq)

    return outs



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)

    ap.add_argument("--outdir", required=True)

    ap.add_argument("--target_bpm", type=float, default=172.0)

    ap.add_argument("--bars", type=int, default=16)

    ap.add_argument("--target_minutes", type=float, default=50.0)

    ap.add_argument("--outro_bars", type=int, default=16)

    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--strict_bpm", action="store_true",
                    help="Stretch every track to the exact target BPM (no +/-2% tolerance)")
    ap.add_argument("--grid", choices=["bar","beat"], default="bar",
                    help="Quantization grid for transitions (bar=default, beat=tighter)")
    ap.add_argument("--use_rubberband", action="store_true")
    ap.add_argument("--harmonic_order", action="store_true")
    ap.add_argument("--order_start", choices=["first","auto"], default="auto")
    ap.add_argument("--cue_csv", action="store_true", help="Write cue CSVs alongside WAVs")
    ap.add_argument("--anchor", choices=["intro","peak","mid","auto"], default="auto",
                    help="Where to derive the track's beat/grid anchor from")
    ap.add_argument("--anchor_skip_bars", type=int, default=16,
                    help="Bars to skip from the start when picking a later anchor")
    ap.add_argument("--stable_min_bars", type=int, default=8,
                    help="Minimum bars of stable groove required when --anchor=auto")
    ap.add_argument("--entry_min_bars", type=int, default=8,
                    help="Earliest bar to consider peaks from the start")
    ap.add_argument("--entry_max_bars", type=int, default=128,
                    help="Latest bar to consider peaks")
    ap.add_argument("--entry_ahead_bars", type=int, default=2,
                    help="How many bars after the crossfade the peak should hit")
    ap.add_argument("--min_tail_bars", type=int, default=8,
                    help="Minimum bars of program outside the crossfade on each side")
    ap.add_argument("--debug_clicks", action="store_true",
                    help="Overlay a metronome click on the mix for phase debugging")
    ap.add_argument("--debug_click_gain", type=float, default=0.12,
                    help="Gain for the debug click track (only when --debug_clicks is set)")
    args = ap.parse_args()



    os.makedirs(args.outdir, exist_ok=True)



    files = []

    for ext in ("*.wav","*.aiff","*.aif","*.flac","*.mp3","*.m4a","*.ogg"):

        files.extend(glob.glob(os.path.join(args.input, ext)))

    files = sorted(files)

    if not files:
        raise SystemExit(f"No audio files found in {args.input}")

    tracks = [analyze_track(p) for p in files]

    for tr in tracks:
        tr.anchor_time = choose_anchor_time(
            tr, target_bpm=args.target_bpm, mode=args.anchor,
            skip_bars=args.anchor_skip_bars, stable_min_bars=args.stable_min_bars
        )

    # Optional harmonic ordering baseline
    order = list(range(len(tracks)))

    if args.harmonic_order and len(tracks) >= 2:

        if args.order_start == "first":

            rest = harmonic_order(tracks[1:], start="auto")

            order = [0] + [i+1 for i in rest]

        else:

            order = harmonic_order(tracks, start="auto")



    # Choose subset to fit target runtime

    subset = select_subset([tracks[i] for i in order], args.target_minutes, args.bars, args.target_bpm)

    subset_idx = [order[i] for i in subset]



    # Build N variants only from the selected subset

    variant_sequences = make_variants(tracks, order, subset_idx, max(1, args.variants))



    print("Analysis:")

    for i,t in enumerate(tracks,1):

        print(f"{i:02d}. {t.name} | {t.bpm:.1f} BPM | {t.key_name} | {t.duration/60:.1f} min")



    for vi, seq in enumerate(variant_sequences, start=1):

        print(f"\nVariant {vi}: " + "  ->  ".join(tracks[i].name for i in seq))

        mix, sr, cues = stitch(
            tracks,
            seq,
            args.target_bpm,
            args.bars,
            args.use_rubberband,
            args.entry_min_bars,
            args.entry_max_bars,
            args.entry_ahead_bars,
            min_tail_bars=args.min_tail_bars,
            debug_clicks=args.debug_clicks,
            debug_click_gain=args.debug_click_gain,
            grid=args.grid,
            strict_bpm=args.strict_bpm,
        )
        mix = enforce_target_length(mix, sr, args.target_minutes, args.outro_bars, args.target_bpm)

        # quick sanity print: how many tracks & final minutes

        print(f" -> Variant {vi} tracks: {len(seq)}, length: {len(mix)/sr/60:.1f} min")

        out_wav = os.path.join(args.outdir, f"mix_v{vi:03d}.wav")

        sf.write(out_wav, mix, sr, subtype="PCM_24")

        if args.cue_csv:

            out_csv = os.path.join(args.outdir, f"mix_v{vi:03d}_cues.csv")

            with open(out_csv, "w", newline="", encoding="utf-8") as f:

                w = csv.writer(f); w.writerow(["Track","StartTimeSeconds","Key","BPM"])

                for (name, t) , idx in zip(cues, seq):

                    tr = tracks[idx]

                    w.writerow([name, f"{t:.3f}", tr.key_name, f"{tr.bpm:.1f}"])



if __name__ == "__main__":

    main()

