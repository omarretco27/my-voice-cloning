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

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
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


def install_colab_dependencies(force: bool = False) -> None:
    """تثبيت كافة متطلبات Habibi-TTS و OpenVoice V2 و Faster-Whisper تلقائياً على Google Colab مع كاش جلسة فوري."""
    marker = Path("/content/.ruwaq_colab_deps_v25_ok")
    if not force and marker.exists() and os.path.exists("/content/Habibi-TTS/src") and os.path.exists("/content/OpenVoice"):
        for p in ("/content/Habibi-TTS/src", "/content/OpenVoice"):
            if p not in sys.path:
                sys.path.insert(0, p)
        return

    def _sh(cmd: str) -> None:
        print(f"⚙️ {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    print("📦 جاري تجهيز محركات الصوت في الاستوديو (تتم مرة واحدة فقط في الجلسة)...")
    _sh("apt-get update -qq && apt-get install -y -qq ffmpeg git libportaudio2")

    if not os.path.exists("/content/Habibi-TTS"):
        _sh("git clone --depth 1 https://github.com/SWivid/Habibi-TTS.git /content/Habibi-TTS")
    if not os.path.exists("/content/OpenVoice"):
        _sh("git clone --depth 1 https://github.com/myshell-ai/OpenVoice.git /content/OpenVoice")

    _sh("pip install -q --upgrade pip wheel setuptools")
    _sh(
        'pip install -q av audioop-lts faster-whisper==1.2.1 edge-tts==7.3.1 '
        'soundfile scipy noisereduce librosa huggingface_hub safetensors '
        'hydra-core omegaconf cached_path tomli "ema_pytorch>=0.5.2" '
        '"x_transformers>=1.31.14" vocos torchdiffeq pydub rjieba '
        'eng_to_ipa inflect unidecode pypinyin cn2an jieba'
    )
    _sh('pip install -q --no-deps "f5-tts>=1.1.5"')

    for p in ("/content/Habibi-TTS/src", "/content/OpenVoice"):
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        import torch
        torch.backends.cudnn.enabled = False
    except Exception:
        pass

    marker.write_text("ok\n", encoding="utf-8")
    print("✅ اكتمل تجهيز محركات الصوت الأساسية بنجاح!")


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
    style_refs_dir = studio_dir / "مراجع_الأداء_والانفعال"
    script_file = studio_dir / "سكريبت_الحلقة.txt"

    for d in (samples_dir, studio_dir, clean_dir, golden_dir, models_dir, outputs_dir, style_refs_dir):
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
        "style_refs_dir": style_refs_dir,
        "script_file": script_file,
    }


def compute_samples_signature(samples_dir: Path) -> tuple[str, list[Path]]:
    """يحسب بصمة سريعة لأسماء وأحجام ملفات «عينة الصوت» لمعرفة ما إذا أضاف عمر عينات جديدة."""
    files = sorted([
        p for p in samples_dir.iterdir()
        if p.is_file() and p.suffix.lower() in prepare_dataset.SUPPORTED_EXTS and not p.name.startswith(".")
    ])
    payload = "|".join(f"{p.name}:{p.stat().st_size}" for p in files)
    sig = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return sig, files


def _ensure_applio_for_rvc_inference(applio_dir: str = "/content/Applio") -> bool:
    """إذا وُجد موديل RVC مدرَّب محفوظ في Google Drive خلال جلسة جديدة، يجهز Applio وملفات RMVPE/ContentVec تلقائياً للاستدلال."""
    try:
        if not os.path.exists(applio_dir):
            print("🔄 تم اكتشاف موديل RVC مدرَّب في Google Drive — جاري تجهيز مشغّل RVC تلقائياً...")
            subprocess.run("apt-get install -y -qq libportaudio2 ffmpeg", shell=True, check=False)
            subprocess.run(f"git clone --depth 1 https://github.com/IAHispano/Applio.git {applio_dir}", shell=True, check=True)
            subprocess.run(
                "pip install -q faiss-cpu soxr einops onnx onnxruntime webrtcvad-wheels resampy stftpitchshift sounddevice",
                shell=True,
                check=True,
            )
            subprocess.run(
                "pip install -q --no-deps torchcrepe torchfcpe local-attention",
                shell=True,
                check=True,
            )

        rmvpe_pt = Path(applio_dir) / "rvc" / "models" / "predictors" / "rmvpe.pt"
        if not rmvpe_pt.exists():
            old_cwd = os.getcwd()
            os.chdir(applio_dir)
            try:
                if applio_dir not in sys.path:
                    sys.path.insert(0, applio_dir)
                import core as applio_core
                print("⬇ جاري تنزيل محدد الطبقة الصوتية (RMVPE + ContentVec) للاستدلال الفوري...")
                applio_core.run_prerequisites_script(pretraineds_hifigan=False, models=True, exe=False)
            finally:
                os.chdir(old_cwd)
        return True
    except Exception as exc:
        print(f"⚠️ تعذّر تجهيز مشغّل RVC تلقائياً ({exc})")
        return False


def sync_and_curate_voice_samples(paths: dict[str, Path], force_rebuild: bool = False) -> dict:
    """
    يدير مكتبة عينات عمر بذكاء:
      1) إذا كانت الحزمة المصفّاة مسبقاً (curated_48k_dataset_v2.zip) مطابقة لملفات «عينة الصوت» الحالية، يفكّها في أقل من ثانية!
      2) إذا أضاف عمر ملفات صوتية جديدة مستقبلاً، يشغّل المدقق الجنائي الذكي (Forensic Gatekeeper) تلقائياً ويحدّث الحزمة.
    """
    samples_dir = paths["samples_dir"]
    clean_dir = paths["clean_dir"]
    golden_dir = paths["golden_dir"]

    current_sig, raw_files = compute_samples_signature(samples_dir)
    if not raw_files:
        raise FileNotFoundError(
            f"لم يتم العثور على ملفات صوتية داخل {samples_dir} — يُرجى وضع عيناتك داخل مجلد «عينة الصوت»."
        )

    meta_file = golden_dir / "speaker_profile_meta.json"
    curated_zip = golden_dir / "curated_48k_dataset_v2.zip"
    habibi_ref_path = golden_dir / "omar_habibi_ref_8s.wav"
    openvoice_ref_path = golden_dir / "omar_openvoice_ref_multi.wav"

    if not force_rebuild and meta_file.exists() and habibi_ref_path.exists():
        try:
            saved_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            saved_meta = {}
        expected_cnt = int(saved_meta.get("clean_segments_count") or 0)
        if (
            saved_meta.get("raw_files_signature") == current_sig
            and saved_meta.get("curation_version") == "2.4-pure-ar"
        ):
            fast_clean = Path("/content/rvc_fast_workspace/clean_48k")
            cur_wavs = list(fast_clean.glob("omar_*_*.wav")) if fast_clean.exists() else []
            if (not cur_wavs or (expected_cnt > 0 and len(cur_wavs) != expected_cnt)) and curated_zip.exists():
                print("⚡ تحميل المقاطع المصفّاة (48kHz PCM_16) إلى القرص السريع المحلي في أقل من ثانية...")
                if fast_clean.exists():
                    shutil.rmtree(fast_clean, ignore_errors=True)
                fast_clean.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(str(curated_zip), "r") as zf:
                    zf.extractall(str(fast_clean))
                cur_wavs = list(fast_clean.glob("omar_*_*.wav"))
            if not cur_wavs:
                cur_wavs = list(clean_dir.glob("omar_*_*.wav"))
            if cur_wavs:
                saved_meta["habibi_ref_wav"] = str(habibi_ref_path)
                saved_meta["openvoice_ref_wav"] = str(openvoice_ref_path)
                saved_meta["local_clean_dir"] = str(fast_clean) if fast_clean.exists() and list(fast_clean.glob("*.wav")) else str(clean_dir)
                print(
                    f"✅ تم تحميل مكتبة صوت عمر المعتمدة كاملة (شاملة عيناتك الجديدة): "
                    f"{saved_meta.get('clean_minutes')} دقيقة صافية "
                    f"({saved_meta.get('clean_segments_count')} مقطع عربي مصري خالص · F0={saved_meta.get('omar_median_f0_hz')}Hz)"
                )
                return saved_meta

    print(f"🎙️ فحص جنائي وتنقية هندسية لـ {len(raw_files)} ملف في «عينة الصوت»...")
    report_path = str(paths["studio_dir"] / "DATASET_AUDIT_REPORT.md")
    res = prepare_dataset.curate_omar_voice_library(
        raw_dir=str(samples_dir),
        clean_dir=str(clean_dir),
        golden_dir=str(golden_dir),
        report_path=report_path,
        min_sec=3.5,
        max_sec=9.2,
    )

    all_segs = sorted(clean_dir.glob("omar_*_*.wav")) or sorted(clean_dir.glob("*.wav"))
    with zipfile.ZipFile(str(curated_zip), "w", zipfile.ZIP_DEFLATED) as zf:
        for seg_p in all_segs:
            zf.write(str(seg_p), arcname=seg_p.name)

    meta = {
        "curation_version": "2.4-pure-ar",
        "raw_files_signature": current_sig,
        "raw_files_count": res["examined_files"],
        "accepted_files_count": res["accepted_files"],
        "rejected_files_count": res["rejected_files"],
        "clean_segments_count": res["segments_count"],
        "tier_counts": res["tier_counts"],
        "tier_minutes": res["tier_minutes"],
        "ar_clean_minutes": res["ar_clean_minutes"],
        "clean_minutes": res["clean_minutes"],
        "omar_median_f0_hz": res["omar_median_f0_hz"],
        "habibi_ref_wav": str(habibi_ref_path),
        "habibi_source_file": res.get("habibi_source_file", ""),
        "habibi_ref_text": "",
        "openvoice_ref_wav": str(openvoice_ref_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"✅ اكتمل الفحص الجنائي: اعتمدنا {res['accepted_files']} ملف أصلي = {res['clean_minutes']} دقيقة صافية "
        f"({res['segments_count']} مقطع · طبقة صوت عمر F0 = {res['omar_median_f0_hz']} Hz)"
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


def _resolve_style_or_guide_file(paths: dict[str, Path], name_or_path: str) -> str | None:
    q = (name_or_path or "").strip()
    if not q:
        return None
    direct = Path(q)
    if direct.exists() and direct.is_file():
        return str(direct)
    cand = paths["style_refs_dir"] / q
    if cand.exists() and cand.is_file():
        return str(cand)
    for p in sorted(paths["style_refs_dir"].iterdir()):
        if p.is_file() and (p.name == q or p.stem == q):
            return str(p)
    print(f"⚠️ لم يُعثر على الملف المرجعي «{q}» داخل {paths['style_refs_dir']} — سيتم استخدام مرجع صوت عمر الأصلي.")
    return None


def run_zero_ui_pipeline(
    drive_root: str = "/content/drive/MyDrive",
    episode_id: str = "EP_AUTO",
    style_or_guide_file: str = "",
    direct_dubbing_mode: bool = False,
    custom_text: str | None = None,
) -> dict:
    """الدالة الرئيسية: بضغطة زر واحدة تقرأ العينات والسكريبت (أو النص المكتوب مباشرة) وتخرج الماستر الجاهز."""
    import riwaq_colab_app as rvc_app

    install_colab_dependencies()
    paths = find_ruwaq_drive_paths(drive_root)
    speaker_meta = sync_and_curate_voice_samples(paths)
    custom_guide = _resolve_style_or_guide_file(paths, style_or_guide_file)

    if custom_text and custom_text.strip():
        script_text = custom_text.strip()
        try:
            paths["script_file"].write_text(script_text + "\n", encoding="utf-8")
        except Exception:
            pass
    else:
        script_text = paths["script_file"].read_text(encoding="utf-8").strip() if paths["script_file"].exists() else ""
        if not script_text:
            script_text = DEFAULT_SCRIPT_AR_EG

    scenes = rvc_app.parse_script_blocks(script_text)
    if not scenes:
        scenes = [{"name": "vo_01", "t_in": 0.2, "text": DEFAULT_SCRIPT_AR_EG}]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = habibi_pipe.CACHE_DIR / ("work_" + ts)
    work_dir.mkdir(parents=True, exist_ok=True)

    use_direct_dub = bool(custom_guide and direct_dubbing_mode)
    if use_direct_dub:
        print(f"🎭 وضع الدوبلاج التمثيلي المباشر: أخذ النطق والانفعال الكامل من «{os.path.basename(custom_guide)}» وتحويله 100% إلى حنجرة عمر!")
        scenes = [{"name": "vo_01", "t_in": 0.2, "text": "[دوبلاج أداء تمثيلي مباشر]"}]
        safe_ref_wav, safe_ref_text = "", ""
    elif custom_guide:
        print(f"🎭 استخدام «{os.path.basename(custom_guide)}» كمرجع خارجي للانفعال واللهجة فقط (مع بقاء الحنجرة 100% بصوت عمر)!")
        tmp_style_48k = str(work_dir / "external_style_ref_48k.wav")
        ext_arr = prepare_dataset.load_audio_mono_48k(custom_guide, 48000)
        sf.write(tmp_style_48k, ext_arr, 48000, subtype="PCM_16")
        safe_ref_wav, safe_ref_text = habibi_pipe.prepare_safe_habibi_reference(tmp_style_48k, "")
        print(f"📌 النص المرجعي المستخلص: «{safe_ref_text.strip()}»")
    else:
        preset_ref_text = speaker_meta.get("habibi_ref_text", "") or ""
        safe_ref_wav, safe_ref_text = habibi_pipe.prepare_safe_habibi_reference(
            speaker_meta["habibi_ref_wav"], preset_ref_text
        )
        print(f"📌 المقطع المرجعي الاستوديو (العامية المصرية): {os.path.basename(safe_ref_wav)}")
        print(f"📌 النص المرجعي: «{safe_ref_text.strip()}»")

    s1_master_concat = []
    s2_master_concat = []
    s1_raw_by_scene: dict[str, str] = {}
    s2_scene_files: list[tuple[str, str]] = []
    pause_48k = np.zeros(int(0.50 * 48000), dtype=np.float32)

    # نستخدم مقاطع الاستوديو العربية النقية فقط (35 مقطع = ~3 دقائق استوديو SNR>50dB) لبصمة OpenVoice V2
    studio_ar_segs = [str(p) for p in sorted(paths["clean_dir"].glob("omar_studio_ar_*.wav"))]
    ov_target_refs = studio_ar_segs if studio_ar_segs else speaker_meta["openvoice_ref_wav"]

    for idx, sc in enumerate(scenes, 1):
        name = sc["name"]
        s1_raw = str(work_dir / f"{name}_s1_raw.wav")
        s2_raw = str(work_dir / f"{name}_s2_raw.wav")
        s1_mst = str(work_dir / f"{name}_s1_48k.wav")
        s2_mst = str(work_dir / f"{name}_s2_48k.wav")

        if use_direct_dub and custom_guide:
            guide_arr = prepare_dataset.load_audio_mono_48k(custom_guide, 48000)
            sf.write(s1_raw, guide_arr, 48000, subtype="PCM_16")
        else:
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
        s1_raw_by_scene[name] = s1_raw

        habibi_pipe.convert_s2_openvoice(
            s1_wav_path=s1_raw,
            target_ref_wav_path=ov_target_refs,
            out_s2_wav=s2_raw,
            tau=0.3,
        )

        voice_finish.finish(s1_raw, s1_mst, target_lufs=-16.0, target_centroid=450.0, sr_out=48000, verbose=False)
        voice_finish.finish(s2_raw, s2_mst, target_lufs=-16.0, target_centroid=445.0, sr_out=48000, verbose=False)

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

    # البحث عن موديل RVC v2 مدرَّب في models/ أو _rvc_workspace/04_models/ وتمرير Habibi-EGY عبره مباشرة!
    out_s3_master = None
    s3_scene_files: list[tuple[str, str]] = []
    alt_exp_dir = paths["studio_dir"] / "_rvc_workspace" / "04_models" / "omar_el_clone_v1"
    pth_path, idx_path, _ = rvc_app.find_trained_rvc_artifacts(str(paths["models_dir"]), model_name="omar_el_clone_v1")
    if not pth_path:
        pth_path, idx_path, _ = rvc_app.find_trained_rvc_artifacts(
            str(paths["studio_dir"] / "_rvc_workspace" / "04_models"),
            model_name="omar_el_clone_v1",
        )

    if pth_path and os.path.exists(pth_path) and _ensure_applio_for_rvc_inference("/content/Applio"):
        try:
            active_clean_dir = speaker_meta.get("local_clean_dir") or str(paths["clean_dir"])
            pure_idx_path, is_pure_ar = rvc_app.ensure_pure_arabic_studio_index(
                clean_dir=active_clean_dir,
                out_model_dir=str(Path(pth_path).parent),
                applio_dir="/content/Applio",
                rvc_exp_dir=str(alt_exp_dir) if alt_exp_dir.exists() else None,
            )
            eff_idx_path = pure_idx_path if is_pure_ar else None
            eff_idx_rate = 0.28 if is_pure_ar else 0.0

            print(
                f"🎛️ وُجد موديل RVC v2 الخاص بعمر ({os.path.basename(pth_path)}) — "
                f"تشغيل وضع «وضوح مخارج الحروف المصرية + قرار عمر 48kHz» "
                f"(protect=0.14 · pure_ar_index={'نعم' if is_pure_ar else 'تلقائي'} · Centroid=490Hz)..."
            )
            rvc_concat = []
            omar_f0 = float(speaker_meta.get("omar_median_f0_hz") or 90.5)
            for sc in scenes:
                name = sc["name"]
                src_habibi = s1_raw_by_scene[name]
                r_wav = str(work_dir / f"{name}_rvc.wav")
                r_crisp = str(work_dir / f"{name}_rvc_crisp.wav")
                r_mst = str(work_dir / f"{name}_rvc_48k.wav")

                s_arr, s_sr = sf.read(src_habibi, dtype="float32")
                src_f0 = prepare_dataset.analyze_signal(s_arr, s_sr)["f0_hz"]
                semi = int(round(12.0 * math.log2(omar_f0 / src_f0))) if src_f0 >= 70.0 else 0
                semi = max(-3, min(3, semi))

                # 1) استنساخ حنجرة عمر مع تمرير الفورمانت الطبيعي خاماً لـ ContentVec (بلا أي تشويه لكاف «إزيّك» أو لام «خلّينا»)
                rvc_app.convert_with_rvc(
                    src_habibi,
                    r_wav,
                    pth_path=pth_path,
                    index_path=eff_idx_path,
                    pitch=semi,
                    index_rate=eff_idx_rate,
                    protect=0.14,
                    volume_envelope=1.0,
                    applio_dir="/content/Applio",
                )

                # 2) استعادة حواف الحروف الساكنة غير المهموسة (3.5–10.2 kHz) بتزامن المللي ثانية
                voice_finish.blend_habibi_consonant_transients(
                    r_wav,
                    src_habibi,
                    r_crisp,
                    sr_out=48000,
                    max_hf_mix=0.22,
                )

                # 3) ماسترينج استوديو عمر الحقيقي (Centroid ~490Hz · -16 LUFS · بدون أي pitch-shift)
                voice_finish.finish(
                    r_crisp,
                    r_mst,
                    target_lufs=-16.0,
                    target_centroid=None,
                    sr_out=48000,
                    clarity_profile="omar_rvc",
                    verbose=False,
                )
                ar, _ = sf.read(r_mst, dtype="float32")
                rvc_concat.extend([ar, pause_48k])
                s3_scene_files.append((name, r_mst))

            out_s3_master = str(out_dir / "03_Omar_RVC_v2_Master_48k.wav")
            sf.write(out_s3_master, np.concatenate(rvc_concat), 48000, subtype="PCM_16")
        except Exception as exc:
            print(f"⚠️ تعذّر توليد نسخة RVC الإضافية ({exc})")

    best_scene_files = s3_scene_files if s3_scene_files else s2_scene_files
    zip_path = str(out_dir / f"{episode_id}_riwaq_pack.zip")
    print("📝 جاري بناء حزمة الحلقة الكاملة مع كابشنز Faster-Whisper...")
    build_riwaq_episode_zip_from_scenes(episode_id, best_scene_files, zip_path, work_dir)

    print("\n" + "=" * 74)
    print("🎉 تم الانتهاء بنجاح! الملفات الصوتية النهائية محفوظة الآن في Google Drive:")
    print("📂 المسار على درايف: رواق / 06-VOICE-CLONING-STUDIO / outputs")
    print(f"  1️⃣ {os.path.basename(out_s1_master)}  (Habibi-TTS المصري الأصلي 48kHz / -16 LUFS)")
    print(f"  2️⃣ {os.path.basename(out_s2_master)}  (Habibi-EGY + بصمة عمر الاستوديو OpenVoice V2)")
    if out_s3_master:
        print(f"  3️⃣ {os.path.basename(out_s3_master)}  (⭐ الذروة الإذاعية: Habibi-EGY + موديل RVC v2 48kHz الخاص بعمر)")
    print(f"  📦 {os.path.basename(zip_path)}  (حزمة الريندر الجاهزة لـ GitHub Actions)")
    print("=" * 74)

    try:
        from IPython.display import Audio, HTML, display
        if out_s3_master:
            display(HTML("<h3>⭐ 1) النسخة الذهبية الأعلى تطابقاً — Habibi المصري + موديل RVC v2 48kHz الخاص بعمر:</h3>"))
            display(Audio(filename=out_s3_master))
        display(HTML("<h3>🎧 2) الاستماع المباشر — Habibi-TTS المصري الأصلي (48kHz / -16 LUFS):</h3>"))
        display(Audio(filename=out_s1_master))
        display(HTML("<h3>🎧 3) الاستماع المباشر — Habibi-EGY + بصمة استوديو عمر المجمّعة OpenVoice V2:</h3>"))
        display(Audio(filename=out_s2_master))
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
    epochs: int = 225,
    episode_id: str = "EP_001",
    reset_from_scratch: bool = False,
    script_text: str | None = None,
) -> dict:
    """خلية تدريب RVC v2 بضغطة واحدة: تستخدم القرص المحلي السريع، تعرض التقدم مباشرة، وتحفظ نقاط الاستئناف في Google Drive ثم تولّد عينة بعد اكتمال التدريب."""
    import riwaq_colab_app as rvc_app

    paths = find_ruwaq_drive_paths(drive_root)
    speaker_meta = sync_and_curate_voice_samples(paths)

    active_clean_dir = Path(speaker_meta.get("local_clean_dir") or paths["clean_dir"])
    clean_wavs = sorted(active_clean_dir.glob("*.wav"))

    lab_compat_root = paths["studio_dir"] / "_rvc_workspace"
    ws = rvc_app.ensure_drive_workspace(str(lab_compat_root))

    print(
        f"🚀 بدء التدريب فائق السرعة (Turbo SSD + GPU RAM) لموديل RVC v2 (48kHz RMVPE) على "
        f"{speaker_meta['clean_minutes']} دقيقة صافية ({len(clean_wavs)} مقطع عربي مصري نقي — شاملة عيناتك الجديدة)..."
    )
    res = rvc_app.train_omar_rvc_model(
        drive_root=str(lab_compat_root),
        applio_dir=applio_dir,
        model_name="omar_el_clone_v1",
        total_epochs=epochs,
        save_every_epoch=75,
        batch_size=8,
        sample_rate=48000,
        dataset_signature=speaker_meta.get("raw_files_signature", ""),
        reset_from_scratch=reset_from_scratch,
        clean_dataset_dir=str(active_clean_dir),
    )
    target_model_dir = paths["models_dir"] / "omar_el_clone_v1"
    target_model_dir.mkdir(parents=True, exist_ok=True)
    if res.get("pth") and os.path.exists(res["pth"]):
        new_pth_name = os.path.basename(res["pth"])
        shutil.copy2(res["pth"], str(target_model_dir / new_pth_name))
        # إزالة النسخ الأقدم من مجلد التصدير لضمان استخدام أحدث موديل دائماً
        for old_p in target_model_dir.glob("*.pth"):
            if old_p.name != new_pth_name and not old_p.name.startswith(("G_", "D_")):
                old_p.unlink(missing_ok=True)
    if res.get("index") and os.path.exists(res["index"]):
        shutil.copy2(res["index"], str(target_model_dir / os.path.basename(res["index"])))
    src_exp_dir = Path(ws["models"]) / "omar_el_clone_v1"
    for extra_name in ("omar_studio_ar_pure.index", "omar_studio_ar_pure.sig"):
        extra_src = src_exp_dir / extra_name
        if extra_src.exists():
            shutil.copy2(str(extra_src), str(target_model_dir / extra_name))

    print("\n🎧 اكتمل التدريب! جاري توليد مقطع تجريبي فوري بموديلك الجديد الآن...")
    gen_res = run_zero_ui_pipeline(drive_root=drive_root, custom_text=script_text, episode_id=episode_id)
    res["generated_preview"] = gen_res
    return res
