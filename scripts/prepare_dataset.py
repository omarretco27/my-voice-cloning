#!/usr/bin/env python3
"""prepare_dataset.py — مدقق ومُجهّز بيانات استنساخ الصوت لمشروع «رِواق»
يقوم بـ:
  1) قراءة أي صيغة صوتية أو فيديو (WAV, MP3, M4A, FLAC, OGG, MP4, MOV, MKV) عبر soundfile/ffmpeg
  2) فحص هندسي لكل ملف (المدة الصافية، التشبع Clipping، نسبة الإشارة للضوضاء SNR، طبقة الصوت F0، مركز الطيف)
  3) تنظيف احترافي لطيف للتدريب الجاف (HPF 65Hz لإزالة همهمة التكييف/المكتب + عزل ضوضاء خفيف عند الحاجة فقط للحفاظ على مخارج السين والصاد)
  4) تقطيع ذكي على الوقفات الطبيعية (3.5 إلى 10 ثوانٍ) بصيغة WAV 48kHz Mono PCM_16 جاهزة لتدريب RVC v2 و XTTS-v2
  5) استخراج أنقى عينة مرجعية تلقائياً (Golden Reference 15–24s) للاستنساخ النصي إن لم يحددها المستخدم يدوياً
  6) إصدار تقرير هندسي بالعربية (DATASET_AUDIT_REPORT.md)
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    from scipy.signal import butter, filtfilt
except ImportError:
    butter = filtfilt = None

try:
    import noisereduce as nr
except ImportError:
    nr = None

SUPPORTED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".mov", ".mkv", ".webm"}
TARGET_SR = 48000


def load_audio_mono_48k(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext in {".wav", ".flac", ".ogg"}:
        try:
            data, sr = sf.read(path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != target_sr:
                n = int(round(len(data) * target_sr / sr))
                data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
            return data
        except Exception:
            pass

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"تعذر قراءة {path}: يُرجى تثبيت ffmpeg لتحويل صيغة {ext}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", path,
            "-vn", "-ac", "1", "-ar", str(target_sr), "-acodec", "pcm_s16le",
            tmp_path,
        ]
        subprocess.run(cmd, check=True)
        data, _ = sf.read(tmp_path, dtype="float32")
        return data
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def highpass_65hz(x: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    x = x - float(np.mean(x))
    if butter is not None and filtfilt is not None and len(x) > 64:
        b, a = butter(2, 65.0 / (sr / 2.0), btype="highpass")
        return filtfilt(b, a, x).astype(np.float32)
    return x.astype(np.float32)


def analyze_signal(x: np.ndarray, sr: int = TARGET_SR) -> dict:
    dur_total = len(x) / sr
    clipped_samples = int(np.sum(np.abs(x) >= 0.995))
    clip_ratio = clipped_samples / max(len(x), 1)

    win = int(0.03 * sr)
    hop = int(0.015 * sr)
    if len(x) < win * 2:
        return {
            "duration_s": round(dur_total, 2),
            "speech_s": round(dur_total, 2),
            "snr_db": 25.0,
            "f0_hz": 0.0,
            "centroid_hz": 0.0,
            "clipped_samples": clipped_samples,
            "clip_ratio": round(clip_ratio, 6),
        }

    rms_list = []
    f0_list = []
    cent_list = []
    hann = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)

    for i in range(0, len(x) - win, hop):
        fr = x[i : i + win].astype(np.float64)
        rms = float(np.sqrt(np.mean(fr ** 2)))
        rms_list.append(rms)
        if rms > 0.008:
            S = np.abs(np.fft.rfft(fr * hann)) ** 2
            cent_list.append(float((S * freqs).sum() / (S.sum() + 1e-12)))
            fr0 = fr - fr.mean()
            ac = np.correlate(fr0, fr0, "full")[len(fr0) - 1 :]
            lo, hi = int(sr / 300), int(sr / 65)
            if hi < len(ac):
                pk = lo + int(np.argmax(ac[lo:hi]))
                if ac[0] > 0 and ac[pk] / ac[0] > 0.32:
                    f0_list.append(sr / pk)

    rms_arr = np.array(rms_list)
    p95 = float(np.percentile(rms_arr, 95)) + 1e-8
    p10 = float(np.percentile(rms_arr, 10)) + 1e-8
    snr_db = float(20.0 * np.log10(max(p95 / p10, 1.0)))

    speech_thr = max(p10 * 2.2, 0.006)
    speech_frames = int(np.sum(rms_arr >= speech_thr))
    speech_s = speech_frames * (hop / sr)

    return {
        "duration_s": round(dur_total, 2),
        "speech_s": round(min(speech_s, dur_total), 2),
        "snr_db": round(snr_db, 1),
        "f0_hz": round(float(np.median(f0_list)) if f0_list else 0.0, 1),
        "centroid_hz": round(float(np.median(cent_list)) if cent_list else 0.0, 1),
        "clipped_samples": clipped_samples,
        "clip_ratio": round(clip_ratio, 6),
    }


def gentle_clean(x: np.ndarray, snr_db: float, sr: int = TARGET_SR, enable_nr: bool = True) -> np.ndarray:
    y = highpass_65hz(x, sr=sr)
    # لا نطبق عزل الضوضاء إلا إذا كانت الضوضاء محسوسة (SNR < 32dB) وبدرجة لطيفة جداً لحماية مخارج الحروف
    if enable_nr and nr is not None and snr_db < 32.0:
        prop = 0.65 if snr_db < 22.0 else 0.45
        y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=prop).astype(np.float32)
    return y


def slice_on_pauses(
    x: np.ndarray,
    sr: int = TARGET_SR,
    min_sec: float = 3.5,
    max_sec: float = 10.0,
    silence_db: float = -40.0,
    min_silence_ms: int = 260,
) -> list:
    """تقطيع الإشارة الصوتية عند الوقفات الطبيعية مع الحفاظ على أطوال مثالية للتدريب."""
    hop_ms = 20
    hop = int(sr * hop_ms / 1000)
    if len(x) <= int(min_sec * sr):
        return [x] if len(x) >= int(1.5 * sr) else []

    rms = np.array([
        float(np.sqrt(np.mean(x[i : i + hop].astype(np.float64) ** 2)) + 1e-9)
        for i in range(0, len(x) - hop, hop)
    ])
    p95 = float(np.percentile(rms, 95))
    thr = max(p95 * (10.0 ** (silence_db / 20.0)), 1e-4)
    is_silent = rms < thr

    # استخراج مناطق الكلام المتصلة
    min_sil_frames = max(1, min_silence_ms // hop_ms)
    regions = []
    in_speech = False
    start_f = 0
    sil_run = 0

    for idx, sil in enumerate(is_silent):
        if not sil:
            if not in_speech:
                start_f = max(0, idx - 4)  # هامش 80ms قبل بداية الكلمة
                in_speech = True
            sil_run = 0
        else:
            if in_speech:
                sil_run += 1
                if sil_run >= min_sil_frames:
                    end_f = idx - sil_run + min(4, sil_run)  # هامش 80ms بعد نهاية الكلمة
                    regions.append((start_f * hop, min(len(x), end_f * hop)))
                    in_speech = False
                    sil_run = 0

    if in_speech:
        regions.append((start_f * hop, len(x)))

    if not regions:
        return []

    # دمج المقاطع القصيرة وتقسيم الطويلة جداً عند أهدأ نقطة
    chunks = []
    cur_s, cur_e = regions[0]
    for s, e in regions[1:]:
        combined_dur = (e - cur_s) / sr
        cur_dur = (cur_e - cur_s) / sr
        gap_dur = (s - cur_e) / sr
        if cur_dur < min_sec and combined_dur <= max_sec and gap_dur < 1.0:
            cur_e = e
        else:
            chunks.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    chunks.append((cur_s, cur_e))

    final_arrays = []
    fade = int(0.020 * sr)
    for s, e in chunks:
        seg = x[s:e]
        # تقسيم آمن إذا تجاوز المقطع max_sec
        while len(seg) > int(max_sec * sr * 1.15):
            search_lo = int(min_sec * sr)
            search_hi = int(max_sec * sr)
            win_step = int(0.02 * sr)
            energies = [
                float(np.mean(seg[p : p + win_step] ** 2))
                for p in range(search_lo, search_hi, win_step)
            ]
            cut = search_lo + int(np.argmin(energies)) * win_step if energies else search_hi
            part = seg[:cut].copy()
            if len(part) >= int(2.0 * sr):
                final_arrays.append(part)
            seg = seg[cut:]
        if len(seg) >= int(2.0 * sr):
            final_arrays.append(seg.copy())

    # تطبيق تلاشي ناعم ومعايرة الذروة إلى -3 dBFS
    target_peak = 10.0 ** (-3.0 / 20.0)
    normalized = []
    for arr in final_arrays:
        mx = float(np.max(np.abs(arr)))
        if mx < 0.005:
            continue
        arr = (arr / mx * target_peak).astype(np.float32)
        if len(arr) > fade * 3:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            arr[:fade] *= ramp
            arr[-fade:] *= ramp[::-1]
        normalized.append(arr)
    return normalized


def pick_golden_reference(chunks_with_scores: list, out_path: str, sr: int = TARGET_SR) -> str | None:
    """دمج أنقى المقاطع لتكوين عينة مرجعية ذهبية (16–24 ثانية) لـ XTTS-v2."""
    if not chunks_with_scores:
        return None
    # ترتيب تنازلي حسب الجودة (SNR وعدم وجود تشبع)
    ranked = sorted(chunks_with_scores, key=lambda item: item[0], reverse=True)
    selected = []
    total_s = 0.0
    pause = np.zeros(int(0.28 * sr), dtype=np.float32)
    for _, arr in ranked:
        dur = len(arr) / sr
        if dur < 3.0 or dur > 12.0:
            continue
        selected.append(arr)
        total_s += dur + 0.28
        if total_s >= 18.0:
            break
    if not selected:
        selected = [ranked[0][1]]

    pieces = []
    for i, seg in enumerate(selected):
        if i > 0:
            pieces.append(pause)
        pieces.append(seg)
    golden = np.concatenate(pieces)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, golden, sr, subtype="PCM_16")
    return out_path


def run_pipeline(
    raw_dir: str,
    clean_dir: str,
    golden_dir: str,
    report_path: str,
    min_sec: float = 3.5,
    max_sec: float = 10.0,
    enable_nr: bool = True,
) -> dict:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise SystemExit(f"✘ مجلد التسجيلات الخام غير موجود: {raw_dir}")

    files = sorted([
        p for p in raw_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith(".")
    ])
    if not files:
        raise SystemExit(
            f"✘ لم يتم العثور على أي ملفات صوتية أو فيديو داخل: {raw_dir}\n"
            f"  الصيغ المدعومة: {', '.join(sorted(SUPPORTED_EXTS))}"
        )

    os.makedirs(clean_dir, exist_ok=True)
    # مسح المقاطع القديمة المُولَّدة مسبقاً لتجنب التكرار
    for old_wav in Path(clean_dir).glob("omar_seg_*.wav"):
        old_wav.unlink()

    audits = []
    all_chunks_scored = []
    seg_idx = 0
    total_raw_s = 0.0
    total_clean_s = 0.0

    print(f"=== فحص وتجهيز تسجيلات رِواق ({len(files)} ملف خام) ===")
    for fp in files:
        try:
            x = load_audio_mono_48k(str(fp), TARGET_SR)
        except Exception as exc:
            print(f"  ⚠ تعذر قراءة {fp.name}: {exc}")
            continue

        stats = analyze_signal(x, TARGET_SR)
        stats["file"] = fp.name
        total_raw_s += stats["duration_s"]

        status = "✅ ممتاز"
        notes = []
        if stats["clip_ratio"] > 0.002:
            status = "⚠️ تشبع بسيط"
            notes.append(f"تشبع {stats['clipped_samples']} عينة")
        if stats["snr_db"] < 16.0:
            status = "⚠️ ضوضاء مرتفعة"
            notes.append(f"SNR منخفض ({stats['snr_db']}dB)")
        stats["status"] = status
        stats["notes"] = " · ".join(notes) if notes else "نقي وجاهز"
        audits.append(stats)

        cleaned = gentle_clean(x, stats["snr_db"], sr=TARGET_SR, enable_nr=enable_nr)
        chunks = slice_on_pauses(cleaned, sr=TARGET_SR, min_sec=min_sec, max_sec=max_sec)
        for ch in chunks:
            seg_idx += 1
            out_wav = os.path.join(clean_dir, f"omar_seg_{seg_idx:04d}.wav")
            sf.write(out_wav, ch, TARGET_SR, subtype="PCM_16")
            dur = len(ch) / TARGET_SR
            total_clean_s += dur
            score = stats["snr_db"] - (stats["clip_ratio"] * 1000.0)
            all_chunks_scored.append((score, ch))

        print(
            f"  {status} {fp.name}: {stats['duration_s']}s خام → {len(chunks)} مقطع"
            f" · F0={stats['f0_hz']}Hz · SNR={stats['snr_db']}dB"
        )

    # فحص أو توليد العينة المرجعية الذهبية (Golden Reference)
    os.makedirs(golden_dir, exist_ok=True)
    existing_refs = [
        p for p in Path(golden_dir).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and p.name != "auto_golden_ref_48k.wav"
    ]
    if existing_refs:
        golden_used = str(existing_refs[0])
        print(f"  ⭐ تم اعتماد العينة المرجعية اليدوية: {existing_refs[0].name}")
    else:
        auto_ref = os.path.join(golden_dir, "auto_golden_ref_48k.wav")
        golden_used = pick_golden_reference(all_chunks_scored, auto_ref, sr=TARGET_SR)
        if golden_used:
            print(f"  ⭐ تم استخراج أنقى عينة مرجعية تلقائياً: {auto_ref}")

    clean_minutes = round(total_clean_s / 60.0, 2)
    verdict = (
        "مثالي لتدريب احترافي كامل (RVC v2 + XTTS)"
        if clean_minutes >= 10.0
        else "مناسب للاستنساخ السريع وتدريب RVC خفيف (يُفضّل إضافة 10 دقائق أخرى لنتيجة إذاعية أقوى)"
    )

    # كتابة التقرير الهندسي بالعربية
    lines = [
        "# تقرير الفحص الهندسي لبيانات صوت «رِواق»",
        "",
        f"- **عدد الملفات الخام المفحوصة:** {len(audits)}",
        f"- **إجمالي المدة الخام:** {round(total_raw_s / 60.0, 2)} دقيقة",
        f"- **إجمالي الصوت الصافي بعد حذف السكتات:** **{clean_minutes} دقيقة** ({seg_idx} مقطع بتردد `48kHz PCM_16`)",
        f"- **العينة المرجعية المعتمدة (Golden Reference):** `{golden_used or 'غير متاح'}`",
        f"- **التقييم الهندسي:** {verdict}",
        "",
        "## تفاصيل الملفات المفحوصة",
        "| الملف | المدة (ث) | الكلام الصافي (ث) | SNR (dB) | طبقة الصوت F0 | مركز الطيف | الحالة | ملاحظات |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in audits:
        lines.append(
            f"| `{a['file']}` | {a['duration_s']} | {a['speech_s']} | {a['snr_db']} | "
            f"{a['f0_hz']}Hz | {a['centroid_hz']}Hz | {a['status']} | {a['notes']} |"
        )

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    summary = {
        "raw_files": len(audits),
        "segments_count": seg_idx,
        "raw_minutes": round(total_raw_s / 60.0, 2),
        "clean_minutes": clean_minutes,
        "golden_reference": golden_used,
        "verdict": verdict,
        "report_path": report_path,
    }
    print(f"\n✔ اكتمل تجهيز البيانات: {clean_minutes} دقيقة صافية ({seg_idx} مقطع) → {clean_dir}")
    print(f"✔ التقرير الهندسي: {report_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="تجهيز وفحص بيانات استنساخ الصوت لمشروع رِواق")
    parser.add_argument("--raw-dir", required=True, help="مجلد التسجيلات الخام (WAV/MP3/M4A/MP4...)")
    parser.add_argument("--clean-dir", required=True, help="مجلد إخراج المقاطع المنظفة 48kHz")
    parser.add_argument("--golden-dir", required=True, help="مجلد العينة المرجعية الذهبية (Golden Reference)")
    parser.add_argument("--report", default="reports/DATASET_AUDIT_REPORT.md", help="مسار تقرير الفحص الهندسي")
    parser.add_argument("--min-sec", type=float, default=3.5, help="الحد الأدنى لطول المقطع بالثواني")
    parser.add_argument("--max-sec", type=float, default=10.0, help="الحد الأقصى لطول المقطع بالثواني")
    parser.add_argument("--no-denoise", action="store_true", help="تعطيل عزل الضوضاء اللطيف")
    parser.add_argument("--json-out", default=None, help="حفظ ملخص JSON اختياري")
    args = parser.parse_args()

    summary = run_pipeline(
        raw_dir=args.raw_dir,
        clean_dir=args.clean_dir,
        golden_dir=args.golden_dir,
        report_path=args.report,
        min_sec=args.min_sec,
        max_sec=args.max_sec,
        enable_nr=not args.no_denoise,
    )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
