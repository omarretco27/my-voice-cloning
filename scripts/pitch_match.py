#!/usr/bin/env python3
"""pitch_match.py — مطابقة جرس الصوت لبصمة المرجع
  1) Phase-vocoder pitch shift (زمن ثابت، نغمة أعلى)
  2) تصحيح الفورمانت (الطابع الدافئ الأصلي يبقى — لا يصبح صوتًا آخر)
  3) يُسلّم للـسلسلة الاستوديو (voice_finish) للمطابقة النهائية
"""
import numpy as np, soundfile as sf, sys, os

SR = 48000
N_FFT = 2048
HOP = 512


def time_stretch(x, stretch):
    """تمديد زمني بواسطة phase vocoder (النغمة ثابتة). stretch>1 = أطول"""
    Hs = int(round(HOP * stretch))
    win = np.hanning(N_FFT)
    out_len = int(len(x) * stretch) + N_FFT * 2
    y = np.zeros(out_len)
    wsum = np.zeros(out_len)
    omega = 2 * np.pi * np.arange(N_FFT // 2 + 1) * HOP / N_FFT
    prev_ph = np.zeros(N_FFT // 2 + 1)
    acc = np.zeros(N_FFT // 2 + 1)
    t = 0
    pos = 0
    first = True
    while t + N_FFT <= len(x) and pos + N_FFT <= out_len:
        fr = x[t:t + N_FFT] * win
        X = np.fft.rfft(fr)
        mag, ph = np.abs(X), np.angle(X)
        if first:
            acc = ph.copy()
            first = False
        else:
            d = ph - prev_ph - omega
            d = np.mod(d + np.pi, 2 * np.pi) - np.pi
            acc = acc + (omega + d) * (Hs / HOP)
        prev_ph = ph
        y[pos:pos + N_FFT] += np.fft.irfft(mag * np.exp(1j * acc), N_FFT) * win
        wsum[pos:pos + N_FFT] += win ** 2
        pos += Hs
        t += HOP
    y = y[:pos + N_FFT]
    wsum = wsum[:pos + N_FFT]
    return y / np.maximum(wsum, 1e-8)


def pitch_shift(x, semitones):
    """إزاحة نغمة مع الحفاظ على المدة: تمديد ثم إعادة معاينة"""
    p = 2 ** (semitones / 12.0)
    y = time_stretch(x, p)                # أطول بـ p — النغمة كما هي
    idx = np.arange(len(x)) * p           # نأخذ كل p عيّنة → المدة الأصلية، النغمة × p
    z = np.interp(idx, np.arange(len(y)), y)
    return z[: len(x)]


def spectral_envelope(x, nfft=4096, hop=1024, smooth_bins=64):
    """مغلّف طيفي طويل المدى (للتصحيح الفورمانتي)"""
    win = np.hanning(nfft)
    acc = None
    cnt = 0
    for t in range(0, max(1, len(x) - nfft), hop):
        X = np.abs(np.fft.rfft(x[t:t + nfft] * win)) + 1e-9
        L = np.log(X)
        acc = L if acc is None else acc + L
        cnt += 1
    env = acc / max(cnt, 1)
    # تنعيم سيفستري: نبقي الترددات المنخفضة (القفصية) فقط
    spec = np.fft.rfft(env)
    spec[smooth_bins:] = 0
    return np.fft.rfft(spec).real if False else np.fft.irfft(spec, len(env))


def formant_preserve(shifted, original, max_db=9.0):
    """يعيد الطابع الفورمانتي الأصلي إلى الصوت المُزاح: EQ ثابت = فرق المغلّفين"""
    env_o = spectral_envelope(original)
    env_s = spectral_envelope(shifted[: len(original)] if len(shifted) >= len(original) else shifted)
    n = min(len(env_o), len(env_s))
    corr = env_o[:n] - env_s[:n]
    corr = np.clip(corr, -max_db, max_db)
    # تنعيم إضافي للمنحنى
    k = 9
    corr = np.convolve(corr, np.ones(k) / k, mode="same")
    # تطبيق كفلتر ثابت في نطاق التردد
    X = np.fft.rfft(shifted)
    f = np.fft.rfftfreq(len(shifted), 1 / SR)
    grid = np.linspace(0, SR / 2, n)
    curve = np.interp(f, grid, corr)
    return np.fft.irfft(X * 10 ** (curve / 20), len(shifted))


def median_f0(x, sr=SR):
    hop = int(0.02 * sr); win = int(0.04 * sr); out = []
    for i in range(0, len(x) - win, hop):
        fr = x[i:i + win]
        if np.sqrt(np.mean(fr ** 2)) < 0.006:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
        lo, hi = int(sr / 320), int(sr / 60)
        if hi >= len(ac):
            continue
        pk = lo + int(np.argmax(ac[lo:hi]))
        if ac[0] > 0 and ac[pk] / ac[0] > 0.30:
            out.append(sr / pk)
    return float(np.median(out)) if out else 0.0


def process(path_in, path_out, semitones=2.5, verbose=True):
    d, sr = sf.read(path_in, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != SR:
        n = int(len(d) * SR / sr)
        idx = np.linspace(0, len(d) - 1, n)
        d = np.interp(idx, np.arange(len(d)), d)
        sr = SR
    f0_before = median_f0(d)
    shifted = pitch_shift(d, semitones)
    pushed = formant_preserve(shifted, d)
    # تنظيف الحواف
    trim = int(0.008 * SR)
    pushed = pushed[trim:-trim] if len(pushed) > 3 * trim else pushed
    f0_after = median_f0(pushed)
    sf.write(path_out, pushed.astype("float32"), SR, subtype="PCM_16")
    if verbose:
        print(f"  {os.path.basename(path_in)}: F0 {f0_before:.0f}→{f0_after:.0f}Hz "
              f"({semitones:+.1f} نصف نغمة · هدف المدة {len(d)/SR:.2f}s → {len(pushed)/SR:.2f}s)")
    return f0_before, f0_after


if __name__ == "__main__":
    src_dir, out_dir, semi = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== Pitch Match (+{semi} semitones · formant-preserved) ===")
    for f in sorted(os.listdir(src_dir)):
        if f.endswith(".wav"):
            process(os.path.join(src_dir, f), os.path.join(out_dir, f), semi)
