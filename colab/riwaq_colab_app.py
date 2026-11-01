#!/usr/bin/env python3
"""riwaq_colab_app.py — المحرك الموحّد وواجهة الاستوديو التفاعلية (Gradio) لمشروع «رِواق» على Google Colab.

يوفر:
  1) إدارة مسارات Google Drive التلقائية وحفظ نقاط التدريب (Checkpoints) دون ضياع عند فصل الجلسة.
  2) تدريب وتحديث موديل البصمة الصوتية (RVC v2 · 48kHz · RMVPE · ContentVec) عبر محرك Applio الحديث.
  3) التوليد المزدوج:
     - المسار النصي (Text → ar-EG Neural أو XTTS-v2 → RVC v2 → سلسلة استوديو رِواق voice_finish.py)
     - المسار الصوتي (تسجيل موبايل سريع أو مسودة voice-01 → RVC v2 → سلسلة استوديو رِواق voice_finish.py)
  4) التغليف الكامل لحلقات «رِواق»: توليد vo_01..vo_N + تفريغ Faster-Whisper كلمة-بكلمة + بناء episode_meta.json + SRT + ملف ZIP جاهز للريندر السحابي.
"""
from __future__ import annotations

import asyncio
import gc
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import soundfile as sf

os.environ.setdefault("COQUI_TOS_AGREED", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import voice_finish
import prepare_dataset
import transcribe_to_meta
import make_srt


# ── إعداد مجلدات Google Drive القياسية لـ «رِواق» ───────────────────────────
def ensure_drive_workspace(drive_root: str = "/content/drive/MyDrive/Riwaq_Voice_Lab") -> dict[str, str]:
    root = Path(drive_root)
    paths = {
        "root": str(root),
        "raw": str(root / "01_raw_recordings"),
        "golden": str(root / "02_golden_reference"),
        "clean": str(root / "03_clean_dataset"),
        "models": str(root / "04_models"),
        "episodes": str(root / "05_episodes_out"),
        "reports": str(root / "06_reports"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    os.environ["APPLIO_LOGS_DIR"] = paths["models"]
    return paths


def find_trained_rvc_artifacts(models_dir: str, model_name: str = "omar_el_clone_v1") -> tuple[str | None, str | None, list[str]]:
    """البحث عن أحدث ملف وزن (.pth) وملف الفهرس (.index) لموديل عمر على Google Drive."""
    mdir = Path(models_dir) / model_name
    if not mdir.exists():
        return None, None, []

    pth_candidates = [
        p for p in mdir.glob("*.pth")
        if not p.name.startswith(("G_", "D_"))
    ]

    def _epoch_key(p: Path):
        m = re.search(r"_(\d+)e", p.stem)
        return (int(m.group(1)) if m else 999999, p.stat().st_mtime)

    pth_sorted = sorted(pth_candidates, key=_epoch_key)
    best_pth = str(pth_sorted[-1]) if pth_sorted else None

    idx_candidates = sorted(mdir.glob("*.index"), key=lambda p: p.stat().st_mtime)
    best_idx = str(idx_candidates[-1]) if idx_candidates else None

    return best_pth, best_idx, [str(p) for p in reversed(pth_sorted)]


def find_golden_reference(golden_dir: str) -> str | None:
    gdir = Path(golden_dir)
    if not gdir.exists():
        return None
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    manual = [p for p in sorted(gdir.iterdir()) if p.is_file() and p.suffix.lower() in exts and p.name != "auto_golden_ref_48k.wav"]
    if manual:
        return str(manual[0])
    auto = gdir / "auto_golden_ref_48k.wav"
    return str(auto) if auto.exists() else None


# ── التكامل مع محرك RVC v2 (Applio) ─────────────────────────────────────────
def _ensure_subprocess_cudnn_guard() -> None:
    """يضمن تعطيل تعارض مكتبات cuDNN الفرعية في العمليات الفرعية التي يطلقها Applio على Colab."""
    site_dir = Path("/tmp/riwaq_site")
    site_dir.mkdir(parents=True, exist_ok=True)
    sc = site_dir / "sitecustomize.py"
    if not sc.exists():
        sc.write_text(
            "try:\n"
            "    import torch\n"
            "    torch.backends.cudnn.enabled = False\n"
            "except Exception:\n"
            "    pass\n",
            encoding="utf-8",
        )
    cur_pp = os.environ.get("PYTHONPATH", "")
    parts = [p for p in cur_pp.split(":") if p]
    if str(site_dir) not in parts:
        os.environ["PYTHONPATH"] = ":".join([str(site_dir)] + parts)


def _call_applio_index(applio_core, model_name: str, algo: str = "Auto") -> None:
    import inspect
    sig = inspect.signature(applio_core.run_index_script)
    if "index_algorithm" in sig.parameters:
        applio_core.run_index_script(model_name=model_name, index_algorithm=algo)
    else:
        applio_core.run_index_script(model_name=model_name, rvc_algorithm=algo)


def _load_applio_core(applio_dir: str = "/content/Applio"):
    adir = Path(applio_dir)
    if not adir.exists():
        raise RuntimeError(
            f"مجلد Applio غير موجود في {applio_dir}. يُرجى تشغيل خلية التجهيز رقم (1) أولاً."
        )
    if str(adir) not in sys.path:
        sys.path.insert(0, str(adir))
    old_cwd = os.getcwd()
    os.chdir(str(adir))
    try:
        import core as applio_core
        return applio_core
    finally:
        os.chdir(old_cwd)


def train_omar_rvc_model(
    drive_root: str = "/content/drive/MyDrive/Riwaq_Voice_Lab",
    applio_dir: str = "/content/Applio",
    model_name: str = "omar_el_clone_v1",
    total_epochs: int = 250,
    save_every_epoch: int = 25,
    batch_size: int = 8,
    sample_rate: int = 48000,
) -> dict:
    """خط التدريب الأوتوماتيكي الكامل مع دعم الاستئناف الذاتي من Google Drive عند انقطاع الجلسة."""
    ws = ensure_drive_workspace(drive_root)
    clean_wavs = list(Path(ws["clean"]).glob("*.wav"))
    if not clean_wavs:
        raise RuntimeError(
            "مجلد المقاطع المنظفة فارغ! شغّل خلية تجهيز وفحص الصوت رقم (2) أولاً."
        )

    model_dir = Path(ws["models"]) / model_name
    has_resume_ckpt = any(model_dir.glob("G_*.pth")) and any(model_dir.glob("D_*.pth"))
    _ensure_subprocess_cudnn_guard()

    old_cwd = os.getcwd()
    os.chdir(applio_dir)
    try:
        if applio_dir not in sys.path:
            sys.path.insert(0, applio_dir)
        import core as applio_core

        print("1/4 ⬇ التأكد من تحميل الموديلات الأساسية المسبقة (HiFi-GAN 48kHz + RMVPE + ContentVec)...")
        applio_core.run_prerequisites_script(pretraineds_hifigan=True, models=True, exe=False)

        if not has_resume_ckpt:
            print(f"2/4 ✂ تجهيز مقاطع التدريب ({len(clean_wavs)} مقطع نقي بتردد {sample_rate}Hz)...")
            applio_core.run_preprocess_script(
                model_name=model_name,
                dataset_path=ws["clean"],
                sample_rate=sample_rate,
                cpu_cores=max(1, (os.cpu_count() or 2)),
                cut_preprocess="Skip",
                process_effects=False,
                chunk_len=3.0,
                overlap_len=0.3,
                normalization_mode="none",
            )

            print("3/4 🧬 استخراج بصمة الطبقة الصوتية (RMVPE) والسمات الصوتية (ContentVec)...")
            applio_core.run_extract_script(
                model_name=model_name,
                f0_method="rmvpe",
                cpu_cores=max(1, (os.cpu_count() or 2)),
                gpu="0",
                sample_rate=sample_rate,
                embedder_model="contentvec",
                embedder_model_custom=None,
                include_mutes=2,
            )

            print("3.5/4 🗂 بناء فهرس البصمة الصوتية المبكر (.index) وحفظه على Google Drive...")
            _call_applio_index(applio_core, model_name=model_name, algo="Auto")
        else:
            print("♻ تم العثور على نقطة حفظ سابقة على Google Drive — سيتم استئناف التدريب مباشرة من آخر Epoch!")

        print(
            f"4/4 🚀 بدء تدريب موديل «{model_name}» حتى {total_epochs} Epoch "
            f"(حفظ تلقائي على Drive كل {save_every_epoch} Epoch)..."
        )
        applio_core.run_train_script(
            model_name=model_name,
            save_every_epoch=save_every_epoch,
            save_only_latest=True,
            save_every_weights=True,
            total_epoch=total_epochs,
            sample_rate=sample_rate,
            batch_size=batch_size,
            gpu="0",
            pretrained=True,
            cleanup=False,
            index_algorithm="Auto",
            cache_data_in_gpu=False,
            custom_pretrained=False,
            g_pretrained_path=None,
            d_pretrained_path=None,
            vocoder="HiFi-GAN",
            checkpointing=False,
        )

        # تحديث الفهرس النهائي إن لزم
        _call_applio_index(applio_core, model_name=model_name, algo="Auto")
    finally:
        os.chdir(old_cwd)

    pth, idx, all_pths = find_trained_rvc_artifacts(ws["models"], model_name=model_name)
    print(f"\n✅ اكتمل التدريب وحُفظ الموديل بأمان على Google Drive!")
    print(f"   • ملف الوزن (.pth)  : {pth}")
    print(f"   • ملف الفهرس (.index): {idx}")
    return {"pth": pth, "index": idx, "all_pths": all_pths}


def convert_with_rvc(
    input_wav: str,
    output_wav: str,
    pth_path: str,
    index_path: str | None = None,
    pitch: int = 0,
    index_rate: float = 0.45,
    protect: float = 0.35,
    applio_dir: str = "/content/Applio",
) -> str:
    _ensure_subprocess_cudnn_guard()
    old_cwd = os.getcwd()
    os.chdir(applio_dir)
    try:
        if applio_dir not in sys.path:
            sys.path.insert(0, applio_dir)
        import core as applio_core

        applio_core.run_infer_script(
            pitch=int(pitch),
            index_rate=float(index_rate) if index_path else 0.0,
            volume_envelope=0.85,
            protect=float(protect),
            f0_method="rmvpe",
            input_path=os.path.abspath(input_wav),
            output_path=os.path.abspath(output_wav),
            pth_path=os.path.abspath(pth_path),
            index_path=os.path.abspath(index_path) if index_path and os.path.exists(index_path) else "",
            f0_autotune=False,
            f0_autotune_strength=1.0,
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
            export_format="WAV",
            embedder_model="contentvec",
        )
    finally:
        os.chdir(old_cwd)
    return output_wav


# ── محركات تحويل النص إلى كلام (TTS: المصري العصبي + XTTS-v2) ─────────────────
_XTTS_CACHE = {"model": None}


def _free_xtts_if_needed():
    if _XTTS_CACHE["model"] is not None:
        del _XTTS_CACHE["model"]
        _XTTS_CACHE["model"] = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def synthesize_edge_ar_eg(text: str, out_wav: str, voice: str = "ar-EG-ShakirNeural", rate_pct: int = 0) -> str:
    import edge_tts

    rate_str = f"{rate_pct:+d}%"
    tmp_mp3 = out_wav + ".tmp.mp3"

    async def _run():
        comm = edge_tts.Communicate(text=text, voice=voice, rate=rate_str)
        await comm.save(tmp_mp3)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(lambda: asyncio.run(_run())).result()
    else:
        asyncio.run(_run())

    audio = prepare_dataset.load_audio_mono_48k(tmp_mp3, 48000)
    sf.write(out_wav, audio, 48000, subtype="PCM_16")
    if os.path.exists(tmp_mp3):
        os.remove(tmp_mp3)
    return out_wav


def synthesize_xtts_ar(text: str, speaker_wav: str, out_wav: str, speed: float = 1.0) -> str:
    import torch
    from TTS.api import TTS

    if _XTTS_CACHE["model"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⏳ تحميل محرك XTTS-v2 على {device}...")
        _XTTS_CACHE["model"] = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    tts = _XTTS_CACHE["model"]
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="ar",
        speed=speed,
        file_path=out_wav,
    )
    return out_wav


# ── تحليل سكريبت المشاهد وتوليد حزمة الحلقة الكاملة لـ «رِواق» ────────────────
def parse_script_blocks(raw_script: str) -> list[dict]:
    """
    يدعم صيغتين سهلتين:
    1) فقرات مفصولة بسطر فارغ → تُسمى تلقائياً vo_01, vo_02, vo_03...
    2) صيغة محددة بالمشهد والتوقيت:
       vo_01 | 0.2 | في زاوية بعيدة من ذاكرة المدن...
       vo_02 | 9.5 | سبعمئة سنة وثلاثة آلاف مخطوطة...
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", raw_script.strip()) if b.strip()]
    parsed = []
    for idx, blk in enumerate(blocks, start=1):
        first_line = blk.splitlines()[0].strip()
        parts = [p.strip() for p in first_line.split("|")]
        if len(parts) >= 3 and re.match(r"^[A-Za-z0-9_\-]+$", parts[0]):
            name = parts[0].replace(".wav", "")
            try:
                t_in = float(parts[1])
            except ValueError:
                t_in = None
            rest_first = "|".join(parts[2:]).strip()
            other_lines = "\n".join(blk.splitlines()[1:]).strip()
            text = f"{rest_first} {other_lines}".strip()
        elif len(parts) == 2 and re.match(r"^[A-Za-z0-9_\-]+$", parts[0]):
            name = parts[0].replace(".wav", "")
            t_in = None
            other_lines = "\n".join(blk.splitlines()[1:]).strip()
            text = f"{parts[1]} {other_lines}".strip()
        else:
            name = f"vo_{idx:02d}"
            t_in = None
            text = " ".join(line.strip() for line in blk.splitlines() if line.strip())
        if text:
            parsed.append({"name": name, "t_in": t_in, "text": text})
    return parsed


def produce_episode_pack(
    project_id: str,
    script_text: str,
    engine_mode: str,
    drive_root: str = "/content/drive/MyDrive/Riwaq_Voice_Lab",
    applio_dir: str = "/content/Applio",
    model_name: str = "omar_el_clone_v1",
    pitch_shift: int = 0,
    index_rate: float = 0.45,
    protect: float = 0.35,
    pace_pct: int = 0,
    target_lufs: float = -16.0,
    run_whisper_captions: bool = True,
) -> tuple[str | None, str | None, str]:
    """توليد كامل لحلقة رِواق: صوت مستنسخ + ماسترينج استوديو + كابشنز + episode_meta.json + ملف ZIP."""
    pid = re.sub(r"[^A-Za-z0-9_\-]", "_", (project_id or "EP_001").strip())
    scenes = parse_script_blocks(script_text)
    if not scenes:
        return None, None, "⚠️ يُرجى كتابة نص السكريبت أولاً."

    ws = ensure_drive_workspace(drive_root)
    pth_path, idx_path, _ = find_trained_rvc_artifacts(ws["models"], model_name=model_name)
    golden_ref = find_golden_reference(ws["golden"])

    needs_rvc = engine_mode in ("ar_eg_plus_rvc", "xtts_plus_rvc")
    needs_xtts = engine_mode in ("xtts_plus_rvc", "xtts_only")

    if needs_rvc and not pth_path:
        return (
            None,
            None,
            "⚠️ لم يتم العثور على موديل RVC مدرَّب بعد في Google Drive! "
            "شغّل خلية التدريب رقم (3) أولاً، أو اختر وضع «XTTS-v2 فوري بعينة مرجعية بدون RVC».",
        )
    if needs_xtts and not golden_ref:
        return (
            None,
            None,
            "⚠️ لم يتم العثور على عينة مرجعية (Golden Reference) في 02_golden_reference! "
            "ارفع مقطعاً بصوتك (15-25 ثانية) أو شغّل خلية تجهيز الصوت رقم (2).",
        )

    ep_root = Path(ws["episodes"]) / pid
    if ep_root.exists():
        shutil.rmtree(ep_root)
    audio_dir = ep_root / "public" / "projects" / pid / "audio"
    data_dir = ep_root / "data" / "projects" / pid
    outputs_dir = ep_root / "outputs"
    reports_dir = ep_root / "reports"
    raw_tmp = ep_root / "_work"
    for d in (audio_dir, data_dir, outputs_dir, reports_dir, raw_tmp):
        d.mkdir(parents=True, exist_ok=True)

    vo_entries_meta = []
    vo_entries_whisper = []
    metrics_rows = []
    cursor_s = 0.2  # بداية أول تعليق بعد 200ms حسب GOLDEN_001
    preview_concat = []

    for sc in scenes:
        name = sc["name"]
        tts_wav = str(raw_tmp / f"{name}_tts.wav")
        rvc_wav = str(raw_tmp / f"{name}_rvc.wav")
        final_wav = str(audio_dir / f"{name}.wav")

        if engine_mode == "ar_eg_plus_rvc":
            synthesize_edge_ar_eg(sc["text"], tts_wav, voice="ar-EG-ShakirNeural", rate_pct=pace_pct)
            convert_with_rvc(
                tts_wav, rvc_wav, pth_path=pth_path, index_path=idx_path,
                pitch=pitch_shift, index_rate=index_rate, protect=protect, applio_dir=applio_dir,
            )
            src_for_master = rvc_wav
        elif engine_mode == "xtts_plus_rvc":
            spd = max(0.75, min(1.3, 1.0 + pace_pct / 100.0))
            synthesize_xtts_ar(sc["text"], golden_ref, tts_wav, speed=spd)
            _free_xtts_if_needed()
            convert_with_rvc(
                tts_wav, rvc_wav, pth_path=pth_path, index_path=idx_path,
                pitch=pitch_shift, index_rate=index_rate, protect=protect, applio_dir=applio_dir,
            )
            src_for_master = rvc_wav
        else:
            spd = max(0.75, min(1.3, 1.0 + pace_pct / 100.0))
            synthesize_xtts_ar(sc["text"], golden_ref, tts_wav, speed=spd)
            src_for_master = tts_wav

        info = voice_finish.finish(
            src_for_master,
            final_wav,
            target_lufs=target_lufs,
            target_centroid=380.0,
            sr_out=48000,
            verbose=True,
        )

        t_in = round(sc["t_in"] if sc["t_in"] is not None else cursor_s, 2)
        dur_s = info["duration_s"]
        cursor_s = round(t_in + dur_s + 0.55, 2)

        vo_entries_meta.append({
            "src": f"projects/{pid}/audio/{name}.wav",
            "t_in": t_in,
            "volume_db": 0,
        })
        vo_entries_whisper.append({"path": final_wav, "t_in": t_in})
        metrics_rows.append((name, t_in, dur_s, info["post_lufs"], info["peak_dbtp"], info["f0_hz"], info["centroid_hz"]))

        seg_audio, _ = sf.read(final_wav, dtype="float32")
        preview_concat.append(seg_audio)
        preview_concat.append(np.zeros(int(0.55 * 48000), dtype="float32"))

    # حفظ معاينة صوتية مجمعة للاستماع الفوري في المتصفح
    full_preview_path = str(ep_root / f"PREVIEW_VO_{pid}.wav")
    if preview_concat:
        sf.write(full_preview_path, np.concatenate(preview_concat), 48000, subtype="PCM_16")

    # بناء ملف episode_meta.json استناداً إلى قالب GOLDEN_001 المعتمد
    template_meta_path = REPO_ROOT / "data" / "projects" / "GOLDEN_001" / "episode_meta.json"
    if template_meta_path.exists():
        meta = json.loads(template_meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"schema_version": "2.0", "brand_ref": {"theme_path": "data/brand.json", "watermark": "on"}}

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

    # نسخ الموسيقى والـ Stingers الافتراضية من GOLDEN_001 لكي يعمل الريندر فوراً بدون ملفات ناقصة
    golden_audio_dir = REPO_ROOT / "public" / "projects" / "GOLDEN_001" / "audio"
    for asset_name in ("music.wav", "stinger_in.wav", "stinger_out.wav"):
        src_asset = golden_audio_dir / asset_name
        if src_asset.exists():
            shutil.copy2(src_asset, audio_dir / asset_name)
    meta["audio"]["stingers"] = [
        {"src": f"projects/{pid}/audio/stinger_in.wav", "t_in": 0.0, "volume_db": -8},
        {"src": f"projects/{pid}/audio/stinger_out.wav", "t_in": max(0.0, round(total_dur - 2.8, 1)), "volume_db": -7},
    ]

    meta_out_path = str(data_dir / "episode_meta.json")
    Path(meta_out_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # تفريغ Faster-Whisper للكابشنز تلقائياً
    caps_note = "تم التخطي حسب الاختيار"
    if run_whisper_captions:
        _free_xtts_if_needed()
        res = transcribe_to_meta.run_episode_transcription(
            vo_entries=vo_entries_whisper,
            project_id=pid,
            out_dir=str(outputs_dir),
            meta_path=meta_out_path,
            model_size="auto",
        )
        caps_note = f"✅ تم توليد {res['captions_count']} كابشن متوافق (≤28 حرفاً · ≥1.25s) + ملف SRT"

    # تقرير القياسات الهندسية للحلقة
    md = [
        f"# تقرير هندسة الصوت لحلقة `{pid}`",
        "",
        f"- **وضع التوليد:** `{engine_mode}`",
        f"- **الكابشنز الآلية:** {caps_note}",
        "",
        "| المقطع | بداية الدخول (ث) | المدة (ث) | الارتفاع LUFS | الذروة dBTP | طبقة F0 | مركز الطيف |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in metrics_rows:
        md.append(f"| `{row[0]}.wav` | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]}Hz | {row[6]}Hz |")
    report_md_text = "\n".join(md)
    (reports_dir / f"VO_METRICS_{pid}.md").write_text(report_md_text + "\n", encoding="utf-8")

    # إزالة ملفات العمل المؤقتة وضغط الحزمة الجاهزة للمستودع
    shutil.rmtree(raw_tmp, ignore_errors=True)
    zip_path = str(Path(ws["episodes"]) / f"{pid}_riwaq_pack.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_p in ep_root.rglob("*"):
            if file_p.is_file() and file_p.name != os.path.basename(full_preview_path):
                zf.write(file_p, file_p.relative_to(ep_root))

    summary_msg = (
        f"### ✅ اكتمل إنتاج حزمة الحلقة `{pid}` بنجاح!\n"
        f"- **مسار الحفظ على Google Drive:** `{ep_root}`\n"
        f"- **ملف التحميل المضغوط الجاهز لـ GitHub:** `{zip_path}`\n\n"
        f"{report_md_text}"
    )
    return full_preview_path, zip_path, summary_msg


def convert_voice_file_to_omar(
    input_audio_path: str,
    drive_root: str = "/content/drive/MyDrive/Riwaq_Voice_Lab",
    applio_dir: str = "/content/Applio",
    model_name: str = "omar_el_clone_v1",
    pitch_shift: int = 0,
    index_rate: float = 0.45,
    protect: float = 0.35,
    target_lufs: float = -16.0,
    keep_silences: bool = False,
) -> tuple[str | None, str | None, str]:
    """تحويل أي تسجيل صوتي (مذكرة صوتية من الموبايل أو صوت معلق آخر) إلى بصمة عمر + ماسترينج رِواق."""
    if not input_audio_path or not os.path.exists(input_audio_path):
        return None, None, "⚠️ يُرجى رفع ملف صوتي أو التسجيل من المايك أولاً."

    ws = ensure_drive_workspace(drive_root)
    pth_path, idx_path, _ = find_trained_rvc_artifacts(ws["models"], model_name=model_name)
    if not pth_path:
        return None, None, "⚠️ لم يتم العثور على موديل RVC مدرَّب بعد! شغّل خلية التدريب رقم (3) أولاً."

    out_folder = Path(ws["episodes"]) / "_voice_to_voice"
    out_folder.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_\-]", "_", Path(input_audio_path).stem) or "recording"
    clean_in = str(out_folder / f"{stem}_48k_in.wav")
    rvc_out = str(out_folder / f"{stem}_rvc_raw.wav")
    master_out = str(out_folder / f"{stem}_omar_master.wav")

    # تنظيف لطيف لتسجيل الموبايل قبل تمريره على RVC
    raw_sig = prepare_dataset.load_audio_mono_48k(input_audio_path, 48000)
    st = prepare_dataset.analyze_signal(raw_sig, 48000)
    cleaned_sig = prepare_dataset.gentle_clean(raw_sig, st["snr_db"], sr=48000, enable_nr=True)
    sf.write(clean_in, cleaned_sig, 48000, subtype="PCM_16")

    convert_with_rvc(
        clean_in, rvc_out, pth_path=pth_path, index_path=idx_path,
        pitch=pitch_shift, index_rate=index_rate, protect=protect, applio_dir=applio_dir,
    )
    info = voice_finish.finish(
        rvc_out, master_out, target_lufs=target_lufs, target_centroid=380.0,
        sr_out=48000, compress_gaps=not keep_silences, verbose=True,
    )

    msg = (
        f"### ✅ تم تحويل التسجيل إلى بصمة صوت عمر بنجاح!\n"
        f"- **الملف الماستر الجاهز (`48kHz PCM_16`):** `{master_out}`\n"
        f"- **القياسات بعد سلسلة استوديو رِواق:** `LUFS {info['post_lufs']}` · "
        f"`Peak {info['peak_dbtp']} dBTP` · `F0 {info['f0_hz']}Hz` · `Centroid {info['centroid_hz']}Hz`"
    )
    return rvc_out, master_out, msg


# ── واجهة الاستوديو الرسومية العربية (Gradio UI) ─────────────────────────────
def launch_riwaq_gradio_studio(
    drive_root: str = "/content/drive/MyDrive/Riwaq_Voice_Lab",
    applio_dir: str = "/content/Applio",
    model_name: str = "omar_el_clone_v1",
    share: bool = True,
):
    import gradio as gr

    ws = ensure_drive_workspace(drive_root)
    pth_path, idx_path, _ = find_trained_rvc_artifacts(ws["models"], model_name=model_name)
    golden_ref = find_golden_reference(ws["golden"])

    status_banner = (
        f"**حالة الاستوديو على Google Drive:**\n"
        f"- موديل بصمة عمر (`RVC v2`): {'✅ جاهز (`' + os.path.basename(pth_path) + '`)' if pth_path else '⏳ لم يُدرَّب بعد (شغّل الخلية 3)'}\n"
        f"- العينة المرجعية الذهبية (`Golden Ref`): {'✅ جاهزة (`' + os.path.basename(golden_ref) + '`)' if golden_ref else '⏳ غير موجودة بعد (شغّل الخلية 2)'}"
    )

    sample_script = (
        "vo_01 | 0.2 | في زاوية بعيدة من ذاكرة المدن، حكاية محدش فكر يفتح دفاترها من سبعمئة سنة.\n\n"
        "vo_02 | 9.5 | أكتر من 73% من المخطوطات القديمة لسه متقريتش، بس كل سطر فيها بيحكي سر جديد.\n\n"
        "vo_03 | 25.5 | الخلاصة: التاريخ مش مجرد أرقام وتواريخ، التاريخ حكايات تستحق اللي يسمعها."
    )

    with gr.Blocks(title="استوديو رِواق الصوتي — Riwaq Voice Studio") as demo:
        gr.Markdown(
            "# 🎙️ استوديو «رِواق» لاستنساخ الصوت وإنتاج الحلقات\n"
            "محرك مزدوج مجاني بالكامل (بديل ElevenLabs) متكامل مع سلسلة استوديو `voice_finish.py` وكابشنز `episode_meta.json`.\n\n"
            + status_banner
        )

        with gr.Tabs():
            # التبويب الأول: من السكريبت إلى حزمة الحلقة
            with gr.TabItem("📝 1) من السكريبت المكتوب إلى حزمة الحلقة (Text ➔ Omar)"):
                with gr.Row():
                    ep_id_in = gr.Textbox(label="معرّف الحلقة (Project ID)", value="EP_001")
                    engine_in = gr.Dropdown(
                        label="محرك النطق والاستنساخ",
                        choices=[
                            ("قارئ العامية المصرية العصبي + بصمة عمر RVC (أدق نطق مصري + خامة صوتك)", "ar_eg_plus_rvc"),
                            ("XTTS-v2 (عينة عمر المرجعية) + بصمة عمر RVC (إيقاعك وخامتك معاً)", "xtts_plus_rvc"),
                            ("XTTS-v2 فوري بالعينة المرجعية فقط (يعمل فوراً حتى قبل تدريب RVC)", "xtts_only"),
                        ],
                        value="ar_eg_plus_rvc" if pth_path else "xtts_only",
                    )

                script_in = gr.Textbox(
                    label="سكريبت المشاهد (افصل كل مشهد بسطر فارغ، أو اكتب: vo_01 | ثانية_البداية | النص)",
                    lines=8,
                    value=sample_script,
                )

                with gr.Accordion("⚙️ إعدادات الهندسة الصوتية المتقدمة (مضبوطة افتراضياً لهوية رِواق)", open=False):
                    with gr.Row():
                        pitch_s1 = gr.Slider(-12, 12, value=0, step=1, label="ضبط الطبقة قبل RVC (0 = طبقتك الطبيعية)")
                        idx_s1 = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="قوة الفهرس Index Rate (0.45 مثالي للعربي)")
                        prot_s1 = gr.Slider(0.0, 0.5, value=0.35, step=0.05, label="حماية الحروف الحلقية والهمس Protect")
                    with gr.Row():
                        pace_s1 = gr.Slider(-20, 25, value=0, step=5, label="سرعة الإلقاء (%)")
                        lufs_s1 = gr.Slider(-18.0, -14.0, value=-16.0, step=0.5, label="الارتفاع المستهدف (LUFS)")
                        whisper_chk = gr.Checkbox(value=True, label="توليد كابشنز Faster-Whisper وتحديث episode_meta.json تلقائياً")

                btn_gen_ep = gr.Button("🚀 توليد حزمة الحلقة الكاملة (صوت ماستر + كابشنز + ZIP)", variant="primary")
                preview_aud_out = gr.Audio(label="🎧 معاينة مجمّعة لتعليق الحلقة (48kHz / -16 LUFS)", type="filepath")
                zip_file_out = gr.File(label="📦 تحميل حزمة الحلقة الجاهزة للمستودع (<ID>_riwaq_pack.zip)")
                ep_report_md = gr.Markdown()

                btn_gen_ep.click(
                    fn=lambda pid, scr, eng, p, ir, pr, pc, lu, wh: produce_episode_pack(
                        project_id=pid,
                        script_text=scr,
                        engine_mode=eng,
                        drive_root=drive_root,
                        applio_dir=applio_dir,
                        model_name=model_name,
                        pitch_shift=p,
                        index_rate=ir,
                        protect=pr,
                        pace_pct=pc,
                        target_lufs=lu,
                        run_whisper_captions=wh,
                    ),
                    inputs=[ep_id_in, script_in, engine_in, pitch_s1, idx_s1, prot_s1, pace_s1, lufs_s1, whisper_chk],
                    outputs=[preview_aud_out, zip_file_out, ep_report_md],
                )

            # التبويب الثاني: من تسجيل موبايل / مسودة صوتية إلى صوت عمر الاستوديو
            with gr.TabItem("🎤 2) من تسجيل موبايل أو مسودة صوتية إلى صوت عمر (Voice ➔ Omar)"):
                gr.Markdown(
                    "سجّل الحلقة بسرعة من موبايلك في أي مكان (أو ارفع ملف مسودة `voice-01`)، "
                    "وسيقوم المحرك بتنظيف التسجيل وتطبيعه ببصمة صوتك الاستوديو ثم تمريره على سلسلة ماسترينج رِواق!"
                )
                voice_in = gr.Audio(label="ارفع ملفاً صوتياً أو سجّل مباشرة من المايك", type="filepath")
                with gr.Row():
                    pitch_s2 = gr.Slider(-12, 12, value=0, step=1, label="فرق الطبقة (0 إذا كان التسجيل بصوتك أنت)")
                    idx_s2 = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="قوة الفهرس Index Rate")
                    prot_s2 = gr.Slider(0.0, 0.5, value=0.35, step=0.05, label="حماية الحروف الساكنة Protect")
                with gr.Row():
                    lufs_s2 = gr.Slider(-18.0, -14.0, value=-16.0, step=0.5, label="الارتفاع المستهدف (LUFS)")
                    keep_sil_chk = gr.Checkbox(value=False, label="الحفاظ على السكتات الأصلية دون ضغط (لمطابقة فيديو مقصوص مسبقاً)")

                btn_conv_v = gr.Button("✨ تحويل إلى صوت عمر الاستوديو (48kHz Master)", variant="primary")
                with gr.Row():
                    rvc_raw_out = gr.Audio(label="خام بعد RVC فقط (قبل الماسترينج)", type="filepath")
                    master_v_out = gr.Audio(label="✅ الماستر النهائي الجاهز للنشر (بعد voice_finish.py)", type="filepath")
                conv_report_md = gr.Markdown()

                btn_conv_v.click(
                    fn=lambda vin, p, ir, pr, lu, ks: convert_voice_file_to_omar(
                        input_audio_path=vin,
                        drive_root=drive_root,
                        applio_dir=applio_dir,
                        model_name=model_name,
                        pitch_shift=p,
                        index_rate=ir,
                        protect=pr,
                        target_lufs=lu,
                        keep_silences=ks,
                    ),
                    inputs=[voice_in, pitch_s2, idx_s2, prot_s2, lufs_s2, keep_sil_chk],
                    outputs=[rvc_raw_out, master_v_out, conv_report_md],
                )

    demo.launch(share=share, inline=True, debug=False)
    return demo
