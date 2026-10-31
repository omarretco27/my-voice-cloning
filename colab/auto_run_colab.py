#!/usr/bin/env python3
"""auto_run_colab.py — التشغيل الآلي الكامل بضغطة واحدة بدون أي واجهات معقدة (مربوط مباشرة بـ Google Drive «رواق»)

يقرأ تلقائياً من مجلد Google Drive الخاص بعمر:
  - العينات الصوتية الخام:  /content/drive/MyDrive/رواق/عينة الصوت/
  - السكريبت المطلوب نطقه: /content/drive/MyDrive/رواق/06-VOICE-CLONING-STUDIO/سكريبت_الحلقة.txt
ويحفظ المخرجات الماستر (48kHz / -16 LUFS) مباشرة في:
  - /content/drive/MyDrive/رواق/06-VOICE-CLONING-STUDIO/outputs/
مع عرض مشغل صوت مباشر داخل الخلية في Colab للاستماع الفوري بدون أي رفع ملفات أو واجهة Gradio!
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
COLAB_DIR = REPO_ROOT / "colab"
for p in (str(SCRIPTS_DIR), str(COLAB_DIR), "/content/Habibi-TTS/src", "/content/OpenVoice"):
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import prepare_dataset
import voice_finish
import transcribe_to_meta
import habibi_egy_pipeline as habibi_pipe


DEFAULT_SCRIPT_AR_EG = (
    "إزيّك، أنا عمر من رواق.\n\n"
    "في ساعات متأخرة من الليل، لما الدنيا بتهدى خالص، الواحد بيبدأ يسمع صوت أفكاره بوضوح لأول مرة.\n\n"
    "خلّينا النهارده نحكي الحكاية بهدوء، واحدة واحدة، من غير استعجال."
)


def find_ruwaq_drive_paths(drive_root: str = "/content/drive/MyDrive") -> dict[str, Path]:
    """يعثر تلقائياً على مجلد «رواق» و«عينة الصوت» على Google Drive مهما كان مسار التركيب."""
    root = Path(drive_root)
    ruwaq_dir = root / "رواق"
    if not ruwaq_dir.exists():
        matches = list(root.glob("**/عينة الصوت"))
        if matches:
            ruwaq_dir = matches[0].parent
        else:
            ruwaq_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = ruwaq_dir / "عينة الصوت"
    studio_dir = ruwaq_dir / "06-VOICE-CLONING-STUDIO"
    clean_dir = studio_dir / "clean_dataset_48k"
    golden_dir = studio_dir / "golden_references"
    models_dir = studio_dir / "models"
    outputs_dir = studio_dir / "outputs"
    script_file = studio_dir / "سكريبت_الحلقة.txt"

    for d in (samples_dir, studio_dir, clean_dir, golden_dir, models_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not script_file.exists():
        script_file.write_text(DEFAULT_SCRIPT_AR_EG + "\n", encoding="utf-8")

    return {
        "ruwaq_dir": ruwaq_dir,
        "samples_dir": samples_dir,
        "studio_dir": studio_dir,
        "clean_dir": clean_dir,
        "golden_dir": golden_dir,
        "models_dir": models_dir,
        "outputs_dir": outputs_dir,
        "script_file": script_file,
    }


def sync_and_curate_voice_samples(paths: dict[str, Path]) -> dict:
    """يفحص كل ملفات الصوت في «عينة الصوت» (بما فيها أي ملفات جديدة يرفعها عمر مستقبلاً) ويجهز المراجع الذهبية."""
    samples_dir = paths["samples_dir"]
    clean_dir = paths["clean_dir"]
    golden_dir = paths["golden_dir"]

    exts = ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac", "*.mp4")
    raw_files = []
    for ext in exts:
        raw_files.extend(sorted(samples_dir.glob(ext)))

    if not raw_files:
        raise FileNotFoundError(
            f"لم يتم العثور على ملفات صوتية داخل {samples_dir} — يُرجى وضع ملفات WAV داخل مجلد «عينة الصوت»."
        )

    print(f"🎙️ تم العثور على {len(raw_files)} ملف صوتي خام في «عينة الصوت» — جاري الفحص والتنقية الهندسية...")
    report_path = str(paths["studio_dir"] / "DATASET_AUDIT_REPORT.md")
    res = prepare_dataset.run_pipeline(
        raw_dir=str(samples_dir),
        clean_dir=str(clean_dir),
        golden_dir=str(golden_dir),
        report_path=report_path,
        min_sec=3.5,
        max_sec=9.5,
        enable_nr=True,
    )

    # اختيار أفضل مرجع لـ Habibi-TTS بطول مثالي (6.5 إلى 9.2 ثانية) لتفادي حد الـ 22 ثانية تماماً
    segs = sorted(clean_dir.glob("omar_seg_*.wav"))
    best_habibi_seg = None
    best_habibi_score = -1.0
    ov_pool = []
    f0_values = []

    for seg in segs:
        data, sr = sf.read(str(seg), dtype="float32")
        dur = len(data) / sr
        rms = float(np.sqrt(np.mean(data ** 2)))
        sig_stats = prepare_dataset.analyze_signal(data, sr)
        f0 = sig_stats["f0_hz"]
        if 65.0 <= f0 <= 180.0:
            f0_values.append(f0)
        if 6.2 <= dur <= 9.2:
            score = sig_stats["snr_db"] * (1.0 - abs(dur - 8.2) / 12.0)
            if score > best_habibi_score:
                best_habibi_score = score
                best_habibi_seg = seg
        if dur >= 4.5:
            ov_pool.append((sig_stats["snr_db"], data, sr))

    habibi_ref_path = golden_dir / "omar_habibi_ref_8s.wav"
    if best_habibi_seg is not None:
        shutil.copy2(str(best_habibi_seg), str(habibi_ref_path))
    else:
        acc = []
        acc_dur = 0.0
        pause = np.zeros(int(0.22 * 48000), dtype=np.float32)
        for seg in segs:
            d, sr = sf.read(str(seg), dtype="float32")
            if acc_dur + len(d) / sr <= 9.0:
                acc.append(d)
                acc.append(pause)
                acc_dur += len(d) / sr + 0.22
            if acc_dur >= 6.8:
                break
        merged = np.concatenate(acc) if acc else sf.read(str(segs[0]), dtype="float32")[0]
        sf.write(str(habibi_ref_path), merged, 48000, subtype="PCM_16")

    # بناء مرجع غني مجمّع (~22–28 ثانية من أنقى المقاطع) لمستخرج البصمة الصوتية في OpenVoice V2
    ov_pool.sort(key=lambda item: item[0], reverse=True)
    ov_parts = []
    ov_dur = 0.0
    pause_48k = np.zeros(int(0.25 * 48000), dtype=np.float32)
    for _, arr, sr in ov_pool[:6]:
        ov_parts.append(arr)
        ov_parts.append(pause_48k)
        ov_dur += len(arr) / sr + 0.25
        if ov_dur >= 24.0:
            break
    openvoice_ref_path = golden_dir / "omar_openvoice_ref_multi.wav"
    if ov_parts:
        sf.write(str(openvoice_ref_path), np.concatenate(ov_parts), 48000, subtype="PCM_16")
    else:
        shutil.copy2(str(habibi_ref_path), str(openvoice_ref_path))

    omar_median_f0 = round(float(np.median(f0_values)), 1) if f0_values else 90.5
    meta = {
        "raw_files_count": len(raw_files),
        "clean_segments_count": len(segs),
        "clean_minutes": res["clean_minutes"],
        "omar_median_f0_hz": omar_median_f0,
        "habibi_ref_wav": str(habibi_ref_path),
        "openvoice_ref_wav": str(openvoice_ref_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (golden_dir / "speaker_profile_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"✅ اكتمل تجهيز البصمة الصوتية لعمر: {res['clean_minutes']} دقيقة صافية "
        f"({len(segs)} مقطع) · طبقة الصوت الطبيعية العميقة F0 = {omar_median_f0} Hz"
    )
    return meta


def build_riwaq_episode_zip_from_scenes(
    project_id: str,
    scene_wavs: list[tuple[str, str]],
    out_zip_path: str,
    work_dir: Path,
) -> str:
    """يبني حزمة ZIP متوافقة 100% مع render-episode.yml على GitHub Actions شاملة الكابشنز المزمنة و episode_meta.json."""
    pid = re.sub(r"[^A-Za-z0-9_\-]", "_", (project_id or "EP_AUTO").strip())
    ep_root = work_dir / f"bundle_{pid}"
    if ep_root.exists():
        shutil.rmtree(ep_root)

    audio_dir = ep_root / "public" / "projects" / pid / "audio"
    data_dir = ep_root / "data" / "projects" / pid
    outputs_dir = ep_root / "outputs"
    for d in (audio_dir, data_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    vo_entries_meta = []
    vo_entries_whisper = []
    cursor_s = 0.2
    for name, mastered_wav in scene_wavs:
        dst_wav = audio_dir / f"{name}.wav"
        shutil.copy2(mastered_wav, str(dst_wav))
        arr, sr = sf.read(str(dst_wav), dtype="float32")
        dur_s = round(len(arr) / sr, 2)
        t_in = round(cursor_s, 2)
        cursor_s = round(t_in + dur_s + 0.55, 2)
        vo_entries_meta.append({"src": f"projects/{pid}/audio/{name}.wav", "t_in": t_in, "volume_db": 0})
        vo_entries_whisper.append({"path": str(dst_wav), "t_in": t_in})

    template_meta_path = REPO_ROOT / "data" / "projects" / "GOLDEN_001" / "episode_meta.json"
    meta = json.loads(template_meta_path.read_text(encoding="utf-8"))
    total_dur = max(15, int(round(cursor_s + 2.5)))
    meta["project"]["id"] = pid
    meta["project"]["title"] = f"حلقة {pid} — رِواق"
    meta["narration"]["mode"] = "provided"
    meta["narration"]["profile_id"] = "omar_el_clone_v1"
    meta["audio"]["vo"] = vo_entries_meta
    meta["audio"]["music"] = {
        "src": f"projects/{pid}/audio/music.wav",
        "t_in": 0,
        "t_out": total_dur,
        "volume_db": -24,
    }
    meta["edit"]["duration_s"] = total_dur
    meta["deliverables"] = {
        "naming_profile": "NAMING.md#v1",
        "master_final": f"exports/MASTER_FINAL_{pid}.mp4",
        "master_web": f"exports/MASTER_WEB_{pid}.mp4",
        "platform_exports": [f"exports/EXPORT_VERTICAL_{pid}.mp4"],
        "captions": [f"outputs/CAPTIONS_AR_{pid}.srt", f"outputs/CAPTIONS_EN_{pid}.srt"],
        "thumbnail": f"exports/THUMB_{pid}_A.png",
        "qc_report": f"reports/QC_{pid}.md",
        "sources_report": f"reports/SOURCES_{pid}.txt",
    }

    golden_audio_dir = REPO_ROOT / "public" / "projects" / "GOLDEN_001" / "audio"
    for asset_name in ("music.wav", "stinger_in.wav", "stinger_out.wav"):
        src_asset = golden_audio_dir / asset_name
        if src_asset.exists():
            shutil.copy2(str(src_asset), str(audio_dir / asset_name))
    meta["audio"]["stingers"] = [
        {"src": f"projects/{pid}/audio/stinger_in.wav", "t_in": 0.0, "volume_db": -8},
        {"src": f"projects/{pid}/audio/stinger_out.wav", "t_in": max(0.0, round(total_dur - 2.8, 1)), "volume_db": -7},
    ]

    meta_out_path = str(data_dir / "episode_meta.json")
    Path(meta_out_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    transcribe_to_meta.run_episode_transcription(
        vo_entries_whisper,
        project_id=pid,
        out_dir=str(outputs_dir),
        meta_path=meta_out_path,
        model_size="auto",
    )

    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_p in ep_root.rglob("*"):
            if file_p.is_file():
                zf.write(str(file_p), arcname=str(file_p.relative_to(ep_root)))
    return out_zip_path


def run_zero_ui_pipeline(drive_root: str = "/content/drive/MyDrive", episode_id: str = "EP_AUTO") -> dict:
    """الدالة الرئيسية: بضغطة زر واحدة تقرأ العينات والسكريبت من Google Drive وتخرج الماستر الجاهز."""
    import riwaq_colab_app as rvc_app

    paths = find_ruwaq_drive_paths(drive_root)
    speaker_meta = sync_and_curate_voice_samples(paths)

    script_text = paths["script_file"].read_text(encoding="utf-8").strip()
    if not script_text:
        script_text = DEFAULT_SCRIPT_AR_EG

    scenes = rvc_app.parse_script_blocks(script_text)
    if not scenes:
        scenes = [{"name": "vo_01", "t_in": 0.2, "text": DEFAULT_SCRIPT_AR_EG}]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = Path("/content/ruwaq_local_cache/work_" + ts)
    work_dir.mkdir(parents=True, exist_ok=True)

    safe_ref_wav, safe_ref_text = habibi_pipe.prepare_safe_habibi_reference(speaker_meta["habibi_ref_wav"], "")
    print(f"📌 المقطع المرجعي المختار تلقائياً (8 ثوانٍ): {os.path.basename(safe_ref_wav)}")
    print(f"📌 النص المرجعي المستخرج آلياً: «{safe_ref_text.strip()}»")

    s1_master_concat = []
    s2_master_concat = []
    s2_scene_files: list[tuple[str, str]] = []
    pause_48k = np.zeros(int(0.50 * 48000), dtype=np.float32)

    for idx, sc in enumerate(scenes, 1):
        name = sc["name"]
        s1_raw = str(work_dir / f"{name}_s1_raw.wav")
        s2_raw = str(work_dir / f"{name}_s2_raw.wav")
        s1_mst = str(work_dir / f"{name}_s1_48k.wav")
        s2_mst = str(work_dir / f"{name}_s2_48k.wav")

        print(f"🎙️ [{idx}/{len(scenes)}] توليد مشهد {name} بالعامية المصرية: {sc['text'][:65]}...")
        habibi_pipe.synthesize_s1_habibi_egy(
            gen_text=sc["text"],
            ref_audio_path=safe_ref_wav,
            ref_text=safe_ref_text,
            out_s1_wav=s1_raw,
            seed=42,
            nfe_step=32,
            speed=0.98,
        )
        habibi_pipe.convert_s2_openvoice(
            s1_wav_path=s1_raw,
            target_ref_wav_path=speaker_meta["openvoice_ref_wav"],
            out_s2_wav=s2_raw,
            tau=0.3,
        )

        voice_finish.finish(s1_raw, s1_mst, target_lufs=-16.0, target_centroid=380.0, sr_out=48000, verbose=False)
        voice_finish.finish(s2_raw, s2_mst, target_lufs=-16.0, target_centroid=380.0, sr_out=48000, verbose=False)

        a1, _ = sf.read(s1_mst, dtype="float32")
        a2, _ = sf.read(s2_mst, dtype="float32")
        s1_master_concat.extend([a1, pause_48k])
        s2_master_concat.extend([a2, pause_48k])
        s2_scene_files.append((name, s2_mst))

    out_dir = paths["outputs_dir"]
    out_s1_master = str(out_dir / "01_Habibi_EGY_Master_48k.wav")
    out_s2_master = str(out_dir / "02_Habibi_OpenVoice_Master_48k.wav")
    sf.write(out_s1_master, np.concatenate(s1_master_concat), 48000, subtype="PCM_16")
    sf.write(out_s2_master, np.concatenate(s2_master_concat), 48000, subtype="PCM_16")

    # إذا كان لدى عمر موديل RVC v2 مدرَّب في models/، نولّد أيضاً النسخة الثالثة تلقائياً!
    out_s3_master = None
    pth_path, idx_path, _ = rvc_app.find_trained_rvc_artifacts(str(paths["models_dir"]), model_name="omar_el_clone_v1")
    if pth_path and os.path.exists(pth_path) and os.path.exists("/content/Applio"):
        try:
            print(f"🎛️ وُجد موديل RVC v2 مدرَّب ({os.path.basename(pth_path)}) — جاري توليد النسخة الثالثة...")
            rvc_concat = []
            for sc in scenes:
                g_wav = str(work_dir / f"{sc['name']}_guide.wav")
                r_wav = str(work_dir / f"{sc['name']}_rvc.wav")
                r_mst = str(work_dir / f"{sc['name']}_rvc_48k.wav")
                rvc_app.synthesize_edge_ar_eg(sc["text"], g_wav, voice="ar-EG-ShakirNeural", rate_pct=-6)
                g_arr, g_sr = sf.read(g_wav, dtype="float32")
                src_f0 = prepare_dataset.analyze_signal(g_arr, g_sr)["f0_hz"]
                omar_f0 = speaker_meta["omar_median_f0_hz"]
                semi = int(round(12.0 * math.log2(omar_f0 / src_f0))) if src_f0 >= 75.0 else -5
                semi = max(-10, min(6, semi))
                rvc_app.convert_with_rvc(
                    g_wav, r_wav, pth_path=pth_path, index_path=idx_path,
                    pitch=semi, index_rate=0.55, protect=0.33, applio_dir="/content/Applio",
                )
                voice_finish.finish(r_wav, r_mst, target_lufs=-16.0, target_centroid=380.0, sr_out=48000, verbose=False)
                ar, _ = sf.read(r_mst, dtype="float32")
                rvc_concat.extend([ar, pause_48k])
            out_s3_master = str(out_dir / "03_Omar_RVC_v2_Master_48k.wav")
            sf.write(out_s3_master, np.concatenate(rvc_concat), 48000, subtype="PCM_16")
        except Exception as exc:
            print(f"⚠️ تعذّر توليد نسخة RVC الإضافية ({exc})")

    zip_path = str(out_dir / f"{episode_id}_riwaq_pack.zip")
    print("📝 جاري بناء حزمة الحلقة الكاملة مع كابشنز Faster-Whisper...")
    build_riwaq_episode_zip_from_scenes(episode_id, s2_scene_files, zip_path, work_dir)

    print("\n" + "=" * 74)
    print("🎉 تم الانتهاء بنجاح! الملفات الصوتية النهائية محفوظة الآن في Google Drive:")
    print(f"📂 المسار على درايف: رواق / 06-VOICE-CLONING-STUDIO / outputs")
    print(f"  1️⃣ {os.path.basename(out_s1_master)}  (Habibi-TTS المصري الأصلي 48kHz / -16 LUFS)")
    print(f"  2️⃣ {os.path.basename(out_s2_master)}  (Habibi-EGY + بصمة عمر المجمّعة OpenVoice V2)")
    if out_s3_master:
        print(f"  3️⃣ {os.path.basename(out_s3_master)}  (نسخة موديل RVC v2 الكامل)")
    print(f"  📦 {os.path.basename(zip_path)}  (حزمة الريندر الجاهزة لـ GitHub Actions)")
    print("=" * 74)

    try:
        from IPython.display import Audio, HTML, display
        display(HTML("<h3>🎧 1) الاستماع المباشر — Habibi-TTS المصري الأصلي (48kHz / -16 LUFS):</h3>"))
        display(Audio(filename=out_s1_master))
        display(HTML("<h3>🎧 2) الاستماع المباشر — Habibi-EGY + بصمة صوت عمر المجمّعة OpenVoice V2:</h3>"))
        display(Audio(filename=out_s2_master))
        if out_s3_master:
            display(HTML("<h3>🎧 3) الاستماع المباشر — موديل RVC v2 الكامل الخاص بعمر:</h3>"))
            display(Audio(filename=out_s3_master))
    except Exception:
        pass

    return {
        "s1_master": out_s1_master,
        "s2_master": out_s2_master,
        "s3_master": out_s3_master,
        "zip_pack": zip_path,
        "outputs_dir": str(out_dir),
        "speaker_meta": speaker_meta,
    }


def train_rvc_v2_one_click(
    drive_root: str = "/content/drive/MyDrive",
    applio_dir: str = "/content/Applio",
    epochs: int = 200,
) -> dict:
    """خلية بضغطة واحدة: تدرّب موديل RVC v2 كامل 48kHz على كل عينات «عينة الصوت» وتحفظه في Google Drive."""
    import riwaq_colab_app as rvc_app

    paths = find_ruwaq_drive_paths(drive_root)
    speaker_meta = sync_and_curate_voice_samples(paths)

    # ربط مجلد العمل بنظام Applio على مساحة Google Drive الـ 100GB
    lab_compat_root = paths["studio_dir"] / "_rvc_workspace"
    ws = rvc_app.ensure_drive_workspace(str(lab_compat_root))
    for old_w in Path(ws["clean"]).glob("*.wav"):
        old_w.unlink()
    for seg in paths["clean_dir"].glob("*.wav"):
        shutil.copy2(str(seg), str(Path(ws["clean"]) / seg.name))

    print(
        f"🚀 بدء تدريب موديل RVC v2 (48kHz RMVPE) على {speaker_meta['clean_minutes']} دقيقة صافية "
        f"من مجلد «عينة الصوت» ({epochs} Epoch)..."
    )
    res = rvc_app.train_omar_rvc_model(
        drive_root=str(lab_compat_root),
        applio_dir=applio_dir,
        model_name="omar_el_clone_v1",
        total_epochs=epochs,
        save_every_epoch=25,
        batch_size=8,
        sample_rate=48000,
    )
    # نسخ الـ .pth والـ .index النهائيين مباشرة إلى 06-VOICE-CLONING-STUDIO/models/omar_el_clone_v1/
    target_model_dir = paths["models_dir"] / "omar_el_clone_v1"
    target_model_dir.mkdir(parents=True, exist_ok=True)
    if res.get("pth") and os.path.exists(res["pth"]):
        shutil.copy2(res["pth"], str(target_model_dir / os.path.basename(res["pth"])))
    if res.get("index") and os.path.exists(res["index"]):
        shutil.copy2(res["index"], str(target_model_dir / os.path.basename(res["index"])))
    return res
