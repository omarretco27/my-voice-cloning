#!/usr/bin/env python3
"""transcribe_to_meta.py — تفريغ صوتي كلمة-بكلمة (Faster-Whisper) وبناء كابشنز «رِواق» تلقائياً
يحترم كافة قواعد الهوية وبوابة الجودة (qc.py + brand.json):
  - بحد أقصى سطران × 28 حرفاً عربياً (إجمالي ≤ 54 حرفاً للكابشن الواحد)
  - مدة بقاء الكابشن ≥ 1.25 ثانية (لتفادي فشل فحص qc.py)
  - إبراز تلقائي للأرقام والنسب وأهم كلمة مؤثرة (≤ كلمتين في keywords)
  - دعم العمل على GPU في Google Colab (large-v3-turbo float16) أو CPU محلياً (small int8)
  - تحديث تلقائي لملف episode_meta.json + توليد CAPTIONS_AR_<ID>.json و CAPTIONS_AR_<ID>.srt
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

AR_STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "اللي", "ده", "دي", "دول", "كان", "كانت", "مش", "ما", "لا", "أو",
    "كل", "بس", "يعني", "كده", "عشان", "علشان", "إن", "أن", "هو", "هي",
    "هم", "احنا", "إحنا", "انت", "أنت", "بين", "بعد", "قبل", "حتى", "لو",
}

NUM_RE = re.compile(r"[0-9٠-٩]+[%٪]?")


def extract_keywords(text: str, max_kw: int = 2) -> list:
    """استخراج الأرقام أو أبرز كلمة دلالية لإبرازها باللون الذهبي في كابشنز رِواق."""
    words = [w.strip("،.؟!«»\"'()[]…:") for w in text.split()]
    words = [w for w in words if w]
    kws = []
    # الأولوية للأرقام والنسب المئوية (قاعدة إلزامية من brand.json)
    for w in words:
        if NUM_RE.search(w) and w not in kws:
            kws.append(w)
            if len(kws) >= max_kw:
                return kws
    # اختيار كلمة مفتاحية مؤثرة واحدة غير شائعة إن لم توجد أرقام
    candidates = [w for w in words if w not in AR_STOPWORDS and len(w) >= 4]
    if candidates and len(kws) < 1:
        best = max(candidates, key=len)
        kws.append(best)
    return kws[:max_kw]


def group_words_to_riwaq_captions(
    words: list,
    offset_s: float = 0.0,
    max_total_chars: int = 52,
    min_dur_s: float = 1.25,
    max_dur_s: float = 3.8,
    pause_split_s: float = 0.42,
) -> list:
    """تجميع الكلمات المزمنة إلى كابشنز متوافقة 100% مع معايير رِواق وبوابة الجودة qc.py."""
    if not words:
        return []

    raw_caps = []
    cur_words = []

    for w in words:
        token = w["word"].strip()
        if not token:
            continue
        w_in = float(w["start"]) + offset_s
        w_out = float(w["end"]) + offset_s

        if not cur_words:
            cur_words.append({"word": token, "start": w_in, "end": w_out})
            continue

        prev_end = cur_words[-1]["end"]
        cand_text = " ".join([x["word"] for x in cur_words] + [token])
        cand_dur = w_out - cur_words[0]["start"]
        gap = w_in - prev_end
        cur_dur = prev_end - cur_words[0]["start"]

        should_split = False
        if len(cand_text) > max_total_chars:
            should_split = True
        elif cand_dur > max_dur_s and cur_dur >= min_dur_s:
            should_split = True
        elif gap >= pause_split_s and cur_dur >= min_dur_s:
            should_split = True
        elif cur_words[-1]["word"].endswith((".", "؟", "!", "…")) and cur_dur >= min_dur_s:
            should_split = True

        if should_split:
            raw_caps.append(cur_words)
            cur_words = [{"word": token, "start": w_in, "end": w_out}]
        else:
            cur_words.append({"word": token, "start": w_in, "end": w_out})

    if cur_words:
        raw_caps.append(cur_words)

    # ضمان أن كل كابشن ≥ min_dur_s (لتمرير فحص qc.py بدون أي FAIL)
    formatted = []
    for idx, grp in enumerate(raw_caps):
        text = " ".join(x["word"] for x in grp).strip()
        t_in = round(grp[0]["start"], 2)
        t_out = round(grp[-1]["end"], 2)

        if t_out - t_in < min_dur_s:
            desired_out = round(t_in + min_dur_s, 2)
            # عدم التداخل مع بداية الكابشن التالي إن وُجد
            if idx + 1 < len(raw_caps):
                next_in = round(raw_caps[idx + 1][0]["start"], 2)
                if desired_out <= next_in - 0.04:
                    t_out = desired_out
                elif formatted and (len(formatted[-1]["ar"]) + 1 + len(text) <= 55):
                    # دمج الكابشن القصير مع السابق إذا كانت المسافة ضيقة
                    formatted[-1]["ar"] = f"{formatted[-1]['ar']} {text}".strip()
                    formatted[-1]["t_out"] = max(round(formatted[-1]["t_in"] + min_dur_s, 2), t_out)
                    formatted[-1]["keywords"] = extract_keywords(formatted[-1]["ar"])
                    continue
                else:
                    t_out = desired_out
            else:
                t_out = desired_out

        formatted.append({
            "t_in": t_in,
            "t_out": t_out,
            "ar": text,
            "en": "",
            "keywords": extract_keywords(text),
        })

    return formatted


def transcribe_file(
    audio_path: str,
    model_size: str = "auto",
    device: str = "auto",
    initial_prompt: str = "بسم الله، أهلاً بيكم في رِواق. حكاية النهارده بالعامية المصرية والعربية الواضحة، مع الترقيم السليم.",
) -> tuple[list, str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("حزمة faster-whisper غير مثبتة. ثبّتها بالأمر: pip install faster-whisper") from exc

    use_cuda = False
    if device == "auto":
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except ImportError:
            use_cuda = False
    else:
        use_cuda = device == "cuda"

    chosen_device = "cuda" if use_cuda else "cpu"
    compute_type = "float16" if use_cuda else "int8"
    if model_size == "auto":
        model_size = "large-v3-turbo" if use_cuda else "small"

    print(f"=== تفريغ Faster-Whisper ({model_size} · {chosen_device}/{compute_type}) ===")
    model = WhisperModel(model_size, device=chosen_device, compute_type=compute_type)
    segments, _ = model.transcribe(
        audio_path,
        language="ar",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 250},
        word_timestamps=True,
        initial_prompt=initial_prompt,
    )

    words = []
    full_text = []
    for seg in segments:
        full_text.append(seg.text.strip())
        for w in seg.words or []:
            words.append({
                "word": w.word.strip(),
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
                "probability": round(float(w.probability), 3),
            })
    return words, " ".join(full_text).strip()


def run_episode_transcription(
    vo_entries: list,
    project_id: str,
    out_dir: str,
    meta_path: str | None = None,
    model_size: str = "auto",
) -> dict:
    """
    vo_entries: قائمة من القواميس مثل [{"path": ".../vo_01.wav", "t_in": 0.2}, ...]
    """
    all_words = []
    all_caps = []
    os.makedirs(out_dir, exist_ok=True)

    for item in vo_entries:
        wav_path = item["path"]
        t_offset = float(item.get("t_in", 0.0))
        words, _ = transcribe_file(wav_path, model_size=model_size)
        shifted_words = [
            {**w, "start": round(w["start"] + t_offset, 3), "end": round(w["end"] + t_offset, 3)}
            for w in words
        ]
        all_words.extend(shifted_words)
        caps = group_words_to_riwaq_captions(words, offset_s=t_offset)
        all_caps.extend(caps)

    words_json_path = os.path.join(out_dir, f"CAPTIONS_AR_{project_id}.json")
    with open(words_json_path, "w", encoding="utf-8") as wf:
        json.dump({"project_id": project_id, "words": all_words, "captions": all_caps}, wf, ensure_ascii=False, indent=2)
    print(f"✔ تم حفظ التوقيتات كلمة-بكلمة: {words_json_path} ({len(all_words)} كلمة · {len(all_caps)} كابشن)")

    if meta_path and os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        meta.setdefault("captions", {})
        meta["captions"].setdefault("styles", ["keyword_highlight"])
        meta["captions"].setdefault("languages", ["ar", "en"])
        meta["captions"]["sync_source"] = "whisper_word_timestamps"
        meta["captions"]["data"] = all_caps
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
        print(f"✔ تم تحديث كابشنز الحلقة تلقائياً في: {meta_path}")

        # توليد ملفات SRT المعتمدة تلقائياً
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import make_srt
        make_srt.build(meta_path, out_dir)

    return {"words_json": words_json_path, "captions_count": len(all_caps), "captions": all_caps}


def main():
    parser = argparse.ArgumentParser(description="تفريغ صوت رِواق إلى كابشنز متوافقة مع episode_meta.json")
    parser.add_argument("audio", nargs="?", help="مسار ملف صوتي مفرد (اختياري إذا مُرر --meta)")
    parser.add_argument("--meta", default=None, help="مسار episode_meta.json لقراءة مسارات VO وتحديث الكابشنز تلقائياً")
    parser.add_argument("--public-dir", default="public", help="مجلد public الأساسي لحل مسارات audio.vo")
    parser.add_argument("--out-dir", default="outputs", help="مجلد حفظ ملفات SRT و JSON")
    parser.add_argument("--offset", type=float, default=0.0, help="إزاحة البداية بالثواني عند تمرير ملف صوتي مفرد")
    parser.add_argument("--project-id", default="EPISODE", help="معرف المشروع عند عدم تمرير --meta")
    parser.add_argument("--model", default="auto", help="حجم موديل Whisper (الافتراضي auto: large-v3-turbo على GPU)")
    args = parser.parse_args()

    if args.meta:
        meta = json.load(open(args.meta, encoding="utf-8"))
        pid = meta["project"]["id"]
        vo_list = []
        for v in meta.get("audio", {}).get("vo", []):
            p = os.path.join(args.public_dir, v["src"])
            if not os.path.exists(p):
                raise SystemExit(f"✘ ملف الصوت المعلن في episode_meta غير موجود: {p}")
            vo_list.append({"path": p, "t_in": float(v.get("t_in", 0.0))})
        run_episode_transcription(vo_list, project_id=pid, out_dir=args.out_dir, meta_path=args.meta, model_size=args.model)
    elif args.audio:
        run_episode_transcription(
            [{"path": args.audio, "t_in": args.offset}],
            project_id=args.project_id,
            out_dir=args.out_dir,
            meta_path=None,
            model_size=args.model,
        )
    else:
        parser.error("يجب تمرير ملف صوتي أو خيار --meta.")


if __name__ == "__main__":
    main()
