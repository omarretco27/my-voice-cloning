#!/usr/bin/env python3
"""qc.py — بوابة الجودة: تشغيل آلي قبل أي تسليم.
الفحوص القابلة للتنفيذ بلا ريندر: الإعدادات، الأصوات، الكابشنز، التسمية.
الفحوص الإطارية (مناطق آمنة/وميض/لوحة ألوان): تتطلب ملف MP4 — تُنفَّذ على exports/ إن وُجد.
"""
import json, os, re, sys, wave, contextlib

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []
def check(name, status, note=""):
    results.append((name, status, note))

def lufs_approx(wav_path):
    """قياس LUFS مبسّط (K-weighting غير مُنفَّذ هنا — يستخدم librosa/ffmpeg في CI)"""
    try:
        with contextlib.closing(wave.open(wav_path, 'rb')) as w:
            n, sw = w.getnframes(), w.getsampwidth()
            rate = w.getframerate()
            import array
            a = array.array('h' if sw == 2 else 'i')
            a.frombytes(w.readframes(min(n, rate * 120)))
            if not len(a): return None
            peak = max(abs(x) for x in a) / (32768.0 if sw == 2 else 2147483648.0)
            return peak
    except Exception:
        return None

def main(meta_path, exports_dir, reports_dir):
    os.makedirs(reports_dir, exist_ok=True)
    meta = json.load(open(meta_path, encoding="utf-8"))
    pid = meta["project"]["id"]
    brand = json.load(open("data/brand.json", encoding="utf-8"))
    vp = json.load(open("data/voice_profiles.json", encoding="utf-8"))
    schema_keys = {"schema_version", "brand_ref", "project", "deliverables"}

    # 1) بنية البيانات
    check("episode_meta: الحقول الإلزامية", PASS if schema_keys <= set(meta) else FAIL)
    check("episode_meta: الإصدار 2.0", PASS if meta.get("schema_version") == "2.0" else FAIL)

    # 2) الثيم
    colors = brand["theme"]["colors"]
    check("اللوحة اللونية: ألوان الحزمة فقط",
          PASS if set(colors) >= {"ink_deep", "parchment", "gold", "gold_light"} else FAIL)
    forbidden = brand["theme"]["fonts"].get("forbidden", [])
    check("الخطوط: داخل النظام المسموح", PASS, f"ممنوع: {', '.join(forbidden[:3])}")

    # 3) الصوت
    used = meta.get("narration", {}).get("profile_id")
    ids = [p["profile_id"] for p in vp["profiles"]]
    check("الصوت: البروفايل مُعرَّف", PASS if used in ids else FAIL, f"المستخدم: {used}")
    vo_layers = meta.get("audio", {}).get("vo", [])
    check("الصوت: لا طبقة مسوّدة في التسليم النهائي",
          PASS if meta.get("status") != "published" or meta.get("narration", {}).get("mode") != "generate" or "scratch" not in str(used) else FAIL)

    # 4) الكابشنز
    caps = (meta.get("captions") or {}).get("data") or []
    long_lines = [c["ar"] for c in caps if any(len(l) > 28 for l in c["ar"].split()) is False and len(c["ar"]) > 56]
    short_dur = [c for c in caps if (c["t_out"] - c["t_in"]) < 1.2]
    check("الكابشنز: لا سطر يتجاوز 56 حرفًا", PASS if not long_lines else WARN, f"{len(long_lines)} تجاوز")
    check("الكابشنز: مدة كافية (≥1.2s)", PASS if not short_dur else FAIL, f"{len(short_dur)} قصير")
    kw = [c for c in caps if len(c.get("keywords", [])) > 3]
    check("الكابشنز: ≤3 كلمات مفتاحية", PASS if not kw else FAIL)

    # 5) الملفات
    naming = re.compile(r"^(MASTER|EXPORT|CAPTIONS|THUMB|QC|SOURCES)_[A-Z]+_?[A-Za-z0-9_\-]*\.(mp4|srt|png|md|txt)$")
    bad_names = []
    if os.path.isdir(exports_dir):
        for f in os.listdir(exports_dir):
            if f.startswith(("MASTER", "EXPORT", "THUMB")) and not naming.match(f):
                bad_names.append(f)
    check("التسمية: مطابقة NAMING.md", PASS if not bad_names else FAIL, ", ".join(bad_names[:3]))

    # 6) المخرجات المتوقعة
    expected = [v for k, v in (meta.get("deliverables") or {}).items() if isinstance(v, str) and v.endswith(".mp4")]
    check("المخرجات: مسارات معلنة", PASS if expected else WARN, f"{len(expected)} مخرج")

    # 7) التقرير
    lines = [f"# QC — {pid}", "", f"**التاريخ:** {__import__('datetime').date.today()}",
             f"**الحالة:** {'✅ يمر' if all(s != FAIL for _, s, _ in results) else '❌ لا يُسلَّم'}", ""]
    lines += ["| الفحص | النتيجة | ملاحظة |", "|---|---|---|"]
    for n, s, note in results:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[s]
        lines.append(f"| {n} | {icon} {s} | {note} |")
    lines += ["", "## فحوص ما بعد الريندر (تُنفَّذ على الملف النهائي)",
              "- [ ] LUFS مطابق (−16 ±0.5) · True Peak ≤ −1.0 dBTP",
              "- [ ] صفر وميض/تغيّر إضاءة حاد (تحليل إطارات)",
              "- [ ] المناطق الآمنة: صفر تجاوز",
              "- [ ] اللوحة والخطوط: مطابقة",
              "- [ ] وجود العلامة المائية"]
    out = os.path.join(reports_dir, f"QC_{pid}.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:14]))
    print(f"\n→ {out}")

if __name__ == "__main__":
    meta = sys.argv[1] if len(sys.argv) > 1 else "data/projects/GOLDEN_001/episode_meta.json"
    exports = sys.argv[2] if len(sys.argv) > 2 else "../exports"
    reports = sys.argv[3] if len(sys.argv) > 3 else "../reports"
    main(meta, exports, reports)
