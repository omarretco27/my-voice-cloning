#!/usr/bin/env python3
"""VO Studio Finish Chain — Riwaq v2.0
مطابقة البصمة الصوتية المرجعية لمشروع «رِواق»:
  HPF 70Hz → Body (+2.2dB@250Hz) → Presence (+1.5dB@1.8kHz) →
  De-harsh (-2.8dB@4.2kHz) → Air-trim (-1.5dB@9kHz) →
  Silence-compress (300ms→190ms) → Soft-compress (2:1) →
  Subtle studio room (5%) → Adaptive Centroid Match (380Hz) →
  Iterative Loudness (-16 LUFS) → True-Peak Limit (-1.0 dBTP)

قاعدة مقفولة في دستور رِواق:
  ممنوع أي إزاحة نغمة (pitch-shift) في سلسلة الماسترينج — بصمة الصوت تأتي طبيعية من الاستنساخ.

أمثلة الاستخدام:
  python scripts/voice_finish.py public/projects/D04/audio/vo_raw.wav -o public/projects/D04/audio/vo_01.wav
  python scripts/voice_finish.py public/projects/D04/audio/vo_01.wav --inplace
  python scripts/voice_finish.py raw_folder/ -o finished_folder/ --sr 48000 --lufs -16
"""
import argparse
import glob
import math
import os
import sys
import numpy as np
import soundfile as sf

try:
    from scipy import signal as sig
    from scipy.signal import lfilter as _scipy_lfilter
except ImportError:
    sig = None
    _scipy_lfilter = None

DEFAULT_SR = 48000


def resample(x: np.ndarray, sr_in: int, sr_out: int = DEFAULT_SR) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n = int(round(len(x) * sr_out / sr_in))
    idx = np.linspace(0, len(x) - 1, n)
    if x.ndim == 1:
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return np.stack(
        [np.interp(idx, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])],
        axis=1,
    ).astype(np.float32)


def biquad(x: np.ndarray, b: list, a: list) -> np.ndarray:
    if _scipy_lfilter is not None:
        return _scipy_lfilter(b, a, x).astype(np.float32)
    y = np.zeros_like(x, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1, b2 = b
    _, a1, a2 = a
    for i, v in enumerate(x):
        fv = float(v)
        o = b0 * fv + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, fv
        y2, y1 = y1, o
        y[i] = o
    return y.astype(np.float32)


def peak_eq(x: np.ndarray, fc: float, gain_db: float, Q: float, sr: int = DEFAULT_SR) -> np.ndarray:
    if abs(gain_db) < 1e-4:
        return x
    A = 10 ** (gain_db / 40.0)
    w = 2 * np.pi * fc / sr
    al = np.sin(w) / (2 * Q)
    c = np.cos(w)
    b = [1 + al * A, -2 * c, 1 - al * A]
    a = [1 + al / A, -2 * c, 1 - al / A]
    return biquad(x, [v / a[0] for v in b], [v / a[0] for v in a])


def high_shelf(x: np.ndarray, fc: float, gain_db: float, S: float = 0.8, sr: int = DEFAULT_SR) -> np.ndarray:
    if abs(gain_db) < 1e-4:
        return x
    A = 10 ** (gain_db / 40.0)
    w = 2 * np.pi * fc / sr
    c = np.cos(w)
    s = np.sin(w)
    al = s / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    t = 2 * np.sqrt(A) * al
    b = [A * ((A + 1) + (A - 1) * c + t), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - t)]
    a = [(A + 1) - (A - 1) * c + t, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - t]
    return biquad(x, [v / a[0] for v in b], [v / a[0] for v in a])


def hpf(x: np.ndarray, fc: float, order: int = 3, sr: int = DEFAULT_SR) -> np.ndarray:
    w = 2 * np.pi * fc / sr
    for _ in range(order):
        al = np.sin(w) / (2 ** 0.5)
        c = np.cos(w)
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        a = [1 + al, -2 * c, 1 - al]
        x = biquad(x, [v / a[0] for v in b], [v / a[0] for v in a])
    return x


def fft_band(x: np.ndarray, lo: float, hi: float, mode: str = "bp", sr: int = DEFAULT_SR) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    if mode == "lp":
        g = 1.0 / (1.0 + (f / hi) ** 2)
    elif mode == "hp":
        g = (f / lo) ** 2 / (1.0 + (f / lo) ** 2)
    else:
        g = 1.0 / (1.0 + ((f - ((lo + hi) / 2.0)) / ((hi - lo) / 1.4)) ** 2)
    return np.fft.irfft(X * g, len(x)).astype(np.float32)


def lufs(x: np.ndarray, sr: int = DEFAULT_SR) -> float:
    """قياس ارتفاع الصوت المتكامل (ITU-R BS.1770 K-weighted LUFS)."""
    y = high_shelf(x, 1500, 4.0, 1.0, sr=sr)
    y = hpf(y, 38, 2, sr=sr)
    bs = int(0.400 * sr)
    st = max(1, int(bs * 0.25))
    if len(y) <= bs:
        m = float(np.mean(y.astype(np.float64) ** 2))
        return -0.691 + 10.0 * np.log10(m + 1e-12)
    blocks = [y[i : i + bs] for i in range(0, len(y) - bs, st)]
    l1 = [-0.691 + 10.0 * np.log10(float(np.mean(b.astype(np.float64) ** 2)) + 1e-12) for b in blocks]
    l1 = [v for v in l1 if v > -70.0]
    if not l1:
        return -70.0
    rel = -0.691 + 10.0 * np.log10(np.mean([10.0 ** ((v + 0.691) / 10.0) for v in l1])) - 10.0
    l2 = [v for v in l1 if v > rel]
    if not l2:
        return -70.0
    return float(-0.691 + 10.0 * np.log10(np.mean([10.0 ** ((v + 0.691) / 10.0) for v in l2])))


def compress(
    x: np.ndarray,
    thr_db: float = -20.0,
    ratio: float = 2.0,
    atk: float = 0.012,
    rel: float = 0.18,
    makeup_db: float = 3.0,
    knee: float = 6.0,
    sr: int = DEFAULT_SR,
) -> np.ndarray:
    env = np.abs(x).astype(np.float64)
    a = float(np.exp(-1.0 / (atk * sr)))
    r = float(np.exp(-1.0 / (rel * sr)))
    e = np.zeros_like(env)
    prev = 0.0
    for i, v in enumerate(env):
        prev = a * prev + (1.0 - a) * v if v > prev else r * prev + (1.0 - r) * v
        e[i] = prev
    e_db = 20.0 * np.log10(e + 1e-9)
    over = e_db - thr_db
    k = np.clip(over / knee + 1.0, 0.0, 2.0) if knee > 0 else (over > 0) * 2.0
    gain_db = np.where(over > 0, -(over * (1.0 - 1.0 / ratio)) * (k / 2.0), 0.0)
    return (x * (10.0 ** ((gain_db + makeup_db) / 20.0))).astype(np.float32)


def room(x: np.ndarray, wet: float = 0.05, sr: int = DEFAULT_SR) -> np.ndarray:
    if wet <= 0.0:
        return x
    ir = np.zeros(int(0.09 * sr), dtype=np.float32)
    for t, g in [(0.006, 0.8), (0.013, 0.6), (0.022, 0.42), (0.034, 0.28), (0.05, 0.18)]:
        idx = int(t * sr)
        if idx < len(ir):
            ir[idx] += g
    ir = fft_band(ir, 200, 5000, "bp", sr=sr)
    mx = float(np.max(np.abs(ir)))
    if mx > 0:
        ir /= mx
    n = len(x) + len(ir) - 1
    N = 1 << (n - 1).bit_length()
    wet_sig = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[: len(x)]
    return (x * (1.0 - wet) + wet_sig * wet).astype(np.float32)


def compress_silences(
    x: np.ndarray,
    thresh_db: float = -42.0,
    max_gap_ms: int = 300,
    target_gap_ms: int = 190,
    sr: int = DEFAULT_SR,
) -> np.ndarray:
    env = np.abs(x)
    win = max(1, int(0.01 * sr))
    if len(x) <= win * 2:
        return x
    frames = [float(np.max(env[i : i + win])) for i in range(0, len(x) - win, win)]
    gaps = []
    thr = 10.0 ** (thresh_db / 20.0)
    run = 0
    for k, v in enumerate(frames):
        if v < thr:
            run += 1
        else:
            if run * 10 > max_gap_ms:
                gaps.append((k - run, k, run * 10))
            run = 0
    if not gaps:
        return x
    out = []
    last = 0
    for s, e, _ in gaps:
        out.append(x[last : s * win])
        keep = int(target_gap_ms / 1000.0 * sr)
        mid = (s * win + e * win) // 2
        half = keep // 2
        seg = x[max(s * win, mid - half) : min(e * win, mid + half)]
        # تلاشي ناعم لمنع أي طقطقة عند ضغط السكتات
        fade = min(int(0.008 * sr), len(seg) // 4)
        if fade > 1:
            seg = seg.copy()
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
        out.append(seg)
        last = e * win
    out.append(x[last:])
    return np.concatenate(out).astype(np.float32)


def _f0(x: np.ndarray, sr: int = DEFAULT_SR) -> float:
    hop = int(0.02 * sr)
    win = int(0.04 * sr)
    out = []
    for i in range(0, len(x) - win, hop):
        fr = x[i : i + win].astype(np.float64)
        if np.sqrt(np.mean(fr ** 2)) < 0.006:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[len(fr) - 1 :]
        lo, hi = int(sr / 320), int(sr / 60)
        if hi >= len(ac):
            continue
        pk = lo + int(np.argmax(ac[lo:hi]))
        if ac[0] > 0 and ac[pk] / ac[0] > 0.3:
            out.append(sr / pk)
    return float(np.median(out)) if out else 0.0


def _centroid(x: np.ndarray, sr: int = DEFAULT_SR) -> float:
    hop = int(0.02 * sr)
    win = int(0.04 * sr)
    cs = []
    hann = np.hanning(win)
    f = np.fft.rfftfreq(win, 1.0 / sr)
    for i in range(0, len(x) - win, hop):
        fr = x[i : i + win].astype(np.float64)
        if np.sqrt(np.mean(fr ** 2)) < 0.006:
            continue
        S = np.abs(np.fft.rfft(fr * hann)) ** 2
        cs.append(float((S * f).sum() / (S.sum() + 1e-12)))
    return float(np.median(cs)) if cs else 0.0


def match_reference(
    x: np.ndarray,
    target_centroid: float = 380.0,
    max_boost_db: float = 4.0,
    step_db: float = 0.6,
    sr: int = DEFAULT_SR,
):
    """مطابقة تكيفية آمنة من الإشارة الأساسية دون تراكم الفلاتر."""
    x_base = x.copy()
    cur = x_base
    boost = 0.0
    for _ in range(8):
        c = _centroid(cur, sr=sr)
        err = target_centroid - c
        if abs(err) < 25.0:
            break
        adj = step_db if err > 0 else -step_db * 0.8
        new_boost = float(max(-2.0, min(max_boost_db, boost + adj)))
        if abs(new_boost - boost) < 1e-4:
            break
        boost = new_boost
        y = peak_eq(x_base, 2400, boost, 0.7, sr=sr)
        cur = high_shelf(y, 6500, boost * 0.55, sr=sr)
    return cur, boost


def true_peak_limit(x: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    c = 10.0 ** (ceiling_db / 20.0)
    mx = float(np.max(np.abs(x)))
    if mx <= c or mx == 0.0:
        return x
    y = x / mx * c * 1.25
    y = np.tanh(y)
    return (y * (c / float(np.max(np.abs(y))))).astype(np.float32)


def condition_habibi_for_rvc(
    path_in: str,
    path_out: str,
    sr_out: int = DEFAULT_SR,
) -> str:
    """
    تهيئة صوتية دقيقة لخرج Habibi-TTS قبل دخوله إلى ContentVec + RMVPE في RVC v2:
    تُبرز فورمانتات الحروف العربية الحلقية والانفجارية (ع، ق، غ، ب، ت، ك، هـ، ء) في نطاق 2–5 kHz
    دون أي تشويه، ليقرأها ContentVec بوضوح تام قبل توليد حبال عمر الصوتية.
    """
    d, sr = sf.read(path_in, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    x = resample(d, sr, sr_out)
    x = x - float(np.mean(x))
    x = hpf(x, 75, 3, sr=sr_out)
    x = peak_eq(x, 320, -1.2, 1.0, sr=sr_out)
    x = peak_eq(x, 2500, +2.4, 0.8, sr=sr_out)
    x = peak_eq(x, 4800, +1.2, 0.9, sr=sr_out)
    x = true_peak_limit(x, -1.5)
    os.makedirs(os.path.dirname(os.path.abspath(path_out)), exist_ok=True)
    sf.write(path_out, x, sr_out, subtype="PCM_16")
    return path_out


def blend_habibi_consonant_transients(
    rvc_wav_path: str,
    habibi_wav_path: str,
    out_wav_path: str,
    sr_out: int = DEFAULT_SR,
    max_hf_mix: float = 0.22,
) -> str:
    """
    يمزج نبضات الحروف الساكنة غير المهموسة عالية التردد (3.5–10.2 kHz مثل ك، ت، س، ش، هـ، ح، خ، ف)
    من أداء Habibi-TTS المصري الأصلي فوق خرج RVC v2 بدقة متزامنة بالمللي ثانية، مع إبقاء
    الحروف المتحركة والرنين الصدري بنسبة 100% من حنجرة عمر (RVC HiFi-GAN) بلا أي ازدواج صوتي.
    """
    r_raw, r_sr = sf.read(rvc_wav_path, dtype="float32")
    h_raw, h_sr = sf.read(habibi_wav_path, dtype="float32")
    if r_raw.ndim > 1:
        r_raw = r_raw.mean(axis=1)
    if h_raw.ndim > 1:
        h_raw = h_raw.mean(axis=1)

    r = resample(r_raw, r_sr, sr_out)
    h = resample(h_raw, h_sr, sr_out)

    if sig is None or len(r) < int(0.15 * sr_out) or len(h) < int(0.15 * sr_out):
        os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)), exist_ok=True)
        sf.write(out_wav_path, r, sr_out, subtype="PCM_16")
        return out_wav_path

    # مطابقة مستوى RMS العام قبل استخلاص النبضات الساكنة
    r_rms = float(np.sqrt(np.mean(r * r) + 1e-12))
    h_rms = float(np.sqrt(np.mean(h * h) + 1e-12))
    h = (h * (r_rms / max(h_rms, 1e-6))).astype(np.float32)

    # محاذاة زمنية دقيقة عبر الارتباط المتبادل لغلاف الطاقة (ضمن نافذة ±80ms)
    env_step = max(1, sr_out // 1000)  # دقة 1ms
    n_min = min(len(r), len(h))
    r_env = np.abs(r[:n_min])[::env_step]
    h_env = np.abs(h[:n_min])[::env_step]
    max_lag_ms = 80
    if len(r_env) > 2 * max_lag_ms + 10:
        best_lag = 0
        best_corr = -1.0
        r_norm = r_env - float(np.mean(r_env))
        h_norm = h_env - float(np.mean(h_env))
        for lag in range(-max_lag_ms, max_lag_ms + 1):
            if lag < 0:
                a, b = r_norm[:lag], h_norm[-lag:]
            elif lag > 0:
                a, b = r_norm[lag:], h_norm[:-lag]
            else:
                a, b = r_norm, h_norm
            if len(a) < 32:
                continue
            c = float(np.dot(a, b) / (math.sqrt(float(np.dot(a, a) * np.dot(b, b))) + 1e-12))
            if c > best_corr:
                best_corr = c
                best_lag = lag
        shift_samples = best_lag * env_step
        if shift_samples > 0:
            h = np.concatenate([np.zeros(shift_samples, dtype=np.float32), h])
        elif shift_samples < 0:
            h = h[-shift_samples:]

    if len(h) < len(r):
        h = np.concatenate([h, np.zeros(len(r) - len(h), dtype=np.float32)])
    else:
        h = h[:len(r)]

    # استخلاص حزمة الحروف الساكنة المهموسة فقط (3500–10200 Hz) بمرشح صفري الطور (Zero-Phase)
    nyq = 0.5 * sr_out
    hi_cut = min(10200.0 / nyq, 0.92)
    lo_cut = 3500.0 / nyq
    sos_hf = sig.butter(4, [lo_cut, hi_cut], btype="bandpass", output="sos")
    sos_lf = sig.butter(3, [85.0 / nyq, 750.0 / nyq], btype="bandpass", output="sos")

    h_hf = sig.sosfiltfilt(sos_hf, h).astype(np.float32)
    r_hf = sig.sosfiltfilt(sos_hf, r).astype(np.float32)
    h_lf = sig.sosfiltfilt(sos_lf, h).astype(np.float32)

    # بوابة ذكية تفتح فقط عند السواكن والانفجارات الصوتية (حيث تعلو طاقة الترددات العالية على الحروف المدّية)
    win = max(1, int(0.008 * sr_out))  # نافذة 8ms ناعمة
    kernel = np.ones(win, dtype=np.float32) / float(win)
    e_hf = np.sqrt(np.convolve(h_hf * h_hf, kernel, mode="same") + 1e-12)
    e_lf = np.sqrt(np.convolve(h_lf * h_lf, kernel, mode="same") + 1e-12)

    ratio = e_hf / (e_lf + 0.004)
    # المدود والحروف الجهرية يكون فيها ratio < 0.15 فتغلق البوابة تماماً (0.0)؛ السواكن تقفز فوق 0.35
    gate = np.clip((ratio - 0.18) / 0.55, 0.0, 1.0)
    smooth_win = max(1, int(0.006 * sr_out))
    gate = np.convolve(gate, np.ones(smooth_win, dtype=np.float32) / float(smooth_win), mode="same")

    # تعويض الفرق الساكن فقط دون مضاعفة ما أنتجه RVC بالفعل
    hf_deficit = (h_hf - 0.55 * r_hf).astype(np.float32)
    blended = (r + (max_hf_mix * gate * hf_deficit)).astype(np.float32)
    blended = true_peak_limit(blended, -1.0)

    os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)), exist_ok=True)
    sf.write(out_wav_path, blended, sr_out, subtype="PCM_16")
    return out_wav_path


def finish(
    path_in: str,
    path_out: str,
    target_lufs: float = -16.0,
    target_centroid: float | None = None,
    sr_out: int = DEFAULT_SR,
    compress_gaps: bool = True,
    clarity_profile: str = "standard",
    verbose: bool = True,
) -> dict:
    d, sr = sf.read(path_in, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    x = resample(d, sr, sr_out)
    x = x - float(np.mean(x))

    pre_cent = _centroid(x, sr=sr_out)
    pre_f0 = _f0(x, sr=sr_out)
    pre_lu = lufs(x, sr=sr_out)

    if clarity_profile == "omar_rvc":
        # معايرة مطابقة لبصمة مايك استوديو عمر الحقيقي (Centroid ~490Hz · وضوح مخارج الحروف المصرية مع دفء الصدر 90Hz)
        eff_centroid = 490.0 if target_centroid is None else float(target_centroid)
        x = hpf(x, 58, 2, sr=sr_out)                         # 1 تنظيف الرومبل تحت 58Hz مع حفظ قرار الصدر 90Hz بنسبة 100%
        x = peak_eq(x, 185, +1.6, 0.9, sr=sr_out)            # 2 دفء الصدر الطبيعي لعمر
        x = peak_eq(x, 340, -1.4, 1.1, sr=sr_out)            # 3 إزالة الكتمة الصندوقية التي تغطي مخارج الحروف
        x = peak_eq(x, 2850, +2.8, 0.85, sr=sr_out)          # 4 إبراز مخارج الحروف المصرية (ع، ق، ك، ب، ت، د، لّ)
        x = peak_eq(x, 5200, +1.5, 1.0, sr=sr_out)           # 5 بلورة السواكن المهموسة (س، ش، هـ، ح، خ، ف)
        x = high_shelf(x, 8500, +0.9, sr=sr_out)             # 6 هواء مايك الاستوديو (Condenser Air 48kHz)
        if compress_gaps:
            x = compress_silences(x, sr=sr_out)              # 7 ضبط فواصل التنفس بسلاسة
        # Attack أبطأ (22ms) يترك الحروف الانفجارية (ب، ت، ك، ق) تعبر بكامل وضوحها دون كبح
        x = compress(x, -20, 1.75, 0.022, 0.18, 2.2, sr=sr_out)
        x = room(x, 0.012, sr=sr_out)                        # 8 جفاف استوديو قريب جداً (1.2% فقط لمنع تلاشي السواكن)
        x, boost = match_reference(x, eff_centroid, max_boost_db=4.5, sr=sr_out)
    else:
        eff_centroid = 380.0 if target_centroid is None else float(target_centroid)
        x = hpf(x, 70, 3, sr=sr_out)                         # 1 تنظيف الترددات المنخفضة (Rumble)
        x = peak_eq(x, 250, +2.2, 0.9, sr=sr_out)            # 2 دفء الصدر (Body)
        x = peak_eq(x, 1800, +1.5, 0.8, sr=sr_out)           # 3 وضوح النطق ومخارج الحروف العربية
        x = peak_eq(x, 4200, -2.8, 1.1, sr=sr_out)           # 4 إزالة الحدة الرقمية (إحساس الـ AI)
        x = high_shelf(x, 9000, -1.5, sr=sr_out)             # 5 تهدئة السطوع الزائد
        if compress_gaps:
            x = compress_silences(x, sr=sr_out)              # 6 ضغط السكتات لإيقاع مشدود
        x = compress(x, -20, 2.0, 0.012, 0.18, 3.0, sr=sr_out)  # 7 ضغط ديناميكي ناعم
        x = room(x, 0.05, sr=sr_out)                         # 8 غرفة استوديو خفيفة جداً (5%)
        x, boost = match_reference(x, eff_centroid, sr=sr_out)  # 8.5 مطابقة مركز الطيف للمرجع

    for _ in range(5):                                   # 9 معايرة الارتفاع المتكامل LUFS
        cur_lu = lufs(x, sr=sr_out)
        if abs(cur_lu - target_lufs) < 0.3:
            break
        g = min(10.0 ** ((target_lufs - cur_lu) / 20.0), 3.5)
        x = (x * g).astype(np.float32)
        if float(np.max(np.abs(x))) > 10.0 ** (-1.0 / 20.0):
            x = true_peak_limit(x, -1.0)

    x = true_peak_limit(x, -1.0)                         # 10 سقف الذروة الحقيقي (-1.0 dBTP)
    os.makedirs(os.path.dirname(os.path.abspath(path_out)), exist_ok=True)
    sf.write(path_out, x, sr_out, subtype="PCM_16")

    post_lu = lufs(x, sr=sr_out)
    post_f0 = _f0(x, sr=sr_out)
    post_cent = _centroid(x, sr=sr_out)
    post_peak = float(20.0 * np.log10(max(float(np.max(np.abs(x))), 1e-9)))

    if verbose:
        print(f"  ✔ {os.path.basename(path_in)} → {path_out} ({sr_out}Hz)")
        print(f"    قبل : LUFS {pre_lu:6.1f} · F0 {pre_f0:5.0f}Hz · Centroid {pre_cent:5.0f}Hz")
        print(
            f"    بعد : LUFS {post_lu:6.1f} · F0 {post_f0:5.0f}Hz · Centroid {post_cent:5.0f}Hz"
            f" · Peak {post_peak:5.2f}dBTP · حضور {boost:+.1f}dB"
        )

    return {
        "path_in": path_in,
        "path_out": path_out,
        "sr": sr_out,
        "duration_s": round(len(x) / sr_out, 2),
        "pre_lufs": round(pre_lu, 2),
        "post_lufs": round(post_lu, 2),
        "f0_hz": round(post_f0, 1),
        "centroid_hz": round(post_cent, 1),
        "peak_dbtp": round(post_peak, 2),
    }


def _default_out_path(path_in: str) -> str:
    d, base = os.path.split(path_in)
    stem, _ = os.path.splitext(base)
    if stem.endswith("_raw"):
        stem = stem[:-4]
    else:
        stem = f"{stem}_master"
    return os.path.join(d or ".", f"{stem}.wav")


def main():
    parser = argparse.ArgumentParser(description="سلسلة استوديو رِواق للماسترينج الصوتي (VO Studio Finish Chain)")
    parser.add_argument("inputs", nargs="+", help="ملف صوتي أو مجلد ملفات صوتية للمعالجة")
    parser.add_argument("-o", "--output", default=None, help="مسار ملف الخرج أو مجلد الخرج")
    parser.add_argument("--inplace", action="store_true", help="الكتابة فوق نفس الملف الأصلي")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="معدل العينة بالهرتز (الافتراضي: 48000 حسب NAMING.md)")
    parser.add_argument("--lufs", type=float, default=-16.0, help="الارتفاع المستهدف بالـ LUFS (الافتراضي: -16.0)")
    parser.add_argument("--centroid", type=float, default=None, help="مركز الطيف المرجعي بالهرتز (تلقائي حسب البروفايل)")
    parser.add_argument(
        "--profile",
        choices=["standard", "omar_rvc"],
        default="standard",
        help="بروفايل الماسترينج: standard أو omar_rvc (وضوح مخارج الحروف لبصمة عمر)",
    )
    parser.add_argument("--keep-silences", action="store_true", help="تعطيل ضغط السكتات الطويلة إذا كان التوقيت مقفولاً مسبقاً")
    args = parser.parse_args()

    print(f"=== سلسلة استوديو رِواق للصوت ({args.profile} · LUFS {args.lufs} · TP -1.0dB · {args.sr}Hz) ===")

    # توسيع المجلدات إن وُجدت
    files = []
    for item in args.inputs:
        if os.path.isdir(item):
            for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"):
                files.extend(sorted(glob.glob(os.path.join(item, ext))))
        elif os.path.exists(item):
            files.append(item)
        else:
            print(f"  ⚠ تخطي (غير موجود): {item}", file=sys.stderr)

    if not files:
        raise SystemExit("✘ لم يتم العثور على ملفات صوتية صالحة للمعالجة.")

    for src in files:
        if args.inplace:
            dst = src
        elif args.output:
            if len(files) > 1 or os.path.isdir(args.output) or args.output.endswith(("/", "\\")):
                os.makedirs(args.output, exist_ok=True)
                stem = os.path.splitext(os.path.basename(src))[0]
                dst = os.path.join(args.output, f"{stem}.wav")
            else:
                dst = args.output
        else:
            dst = _default_out_path(src)

        finish(
            src,
            dst,
            target_lufs=args.lufs,
            target_centroid=args.centroid,
            sr_out=args.sr,
            compress_gaps=not args.keep_silences,
            clarity_profile=args.profile,
            verbose=True,
        )


if __name__ == "__main__":
    main()
