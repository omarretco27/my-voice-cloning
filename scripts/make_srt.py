#!/usr/bin/env python3
"""make_srt.py — توليد كابشنز AR/EN من episode_meta.json
القواعد: سطران أقصى · ≤28 حرفًا عربيًا للسطر · عزل النص اللاتيني (BiDi)
"""
import json, os, re, sys

MAX_CHARS = 28
MAX_LINES = 2

def ts(sec: float) -> str:
    ms = int(round((sec - int(sec)) * 1000)); s = int(sec)
    m, s = divmod(s, 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def wrap_ar(text: str, max_chars: int = MAX_CHARS, max_lines: int = MAX_LINES):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) <= max_chars or not cur:
            cur = cand
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    if len(lines) > max_lines:  # دمج الزائد في السطر الأخير (بدل فقد النص)
        lines = lines[:max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return "\n".join(lines)

def isolate_latin(text: str) -> str:
    # عزل المقاطع اللاتينية بعلامات اتجاه (Unicode BiDi isolates)
    return re.sub(r"([A-Za-z][A-Za-z0-9 .&\-]*)", "\u2066\\1\u2069", text)

def build(meta_path: str, out_dir: str):
    meta = json.load(open(meta_path, encoding="utf-8"))
    pid = meta["project"]["id"]
    lines = (meta.get("captions") or {}).get("data") or []
    if not lines:
        print("لا كابشنز في episode_meta.json"); return
    os.makedirs(out_dir, exist_ok=True)
    ar_path = os.path.join(out_dir, f"CAPTIONS_AR_{pid}.srt")
    en_path = os.path.join(out_dir, f"CAPTIONS_EN_{pid}.srt")

    warn = []
    with open(ar_path, "w", encoding="utf-8") as fa, open(en_path, "w", encoding="utf-8") as fe:
        ia = ie = 0
        for ln in lines:
            dur = ln["t_out"] - ln["t_in"]
            if dur < 1.2:
                warn.append(f"مدة قصيرة ({dur:.2f}s): {ln['ar'][:24]}…")
            if ln.get("ar"):
                ia += 1
                fa.write(f"{ia}\n{ts(ln['t_in'])} --> {ts(ln['t_out'])}\n{isolate_latin(wrap_ar(ln['ar']))}\n\n")
            if ln.get("en"):
                ie += 1
                fe.write(f"{ie}\n{ts(ln['t_in'])} --> {ts(ln['t_out'])}\n{ln['en']}\n\n")
    print(f"✔ {ar_path} ({ia}) · {en_path} ({ie})")
    for w in warn[:8]:
        print("  ⚠", w)

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/projects/GOLDEN_001/episode_meta.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    build(src, out)
