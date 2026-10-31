#!/usr/bin/env python3
"""habibi_egy_pipeline.py — استوديو «رِواق» الخفيف للعامية المصرية (Habibi-TTS Specialized/EGY + OpenVoice V2)

مُصمَّم خصيصاً لحل جميع عوائق تقرير VL-NCR-002 / CA-02 دفعة واحدة:
  1) صفر استهلاك من مساحة Google Drive (0 MB): يحمّل الأوزان في قرص Colab المؤقت (/content/models_cache - 100GB)
     علاجاً لمشكلة امتلاء درايف بنسبة 91% (13.68GB / 15GB).
  2) إصلاح عيب CA-02-D01 جذرياً: يستخدم حصرياً أوزان Specialized/EGY بترخيص Apache-2.0 التجاري مع
     dialect_id=None (بدون رموز ⑥〈…〉 الخاصة بـ Unified) وفحص مسبق لخلو النص من الرموز المجهولة (No-OOV).
  3) ترقيع توافق OpenVoice V2 مع Python 3.12 و PyTorch 2.6+ و librosa 0.11+ (إصلاح return_complex=True و librosa.filters.mel).
  4) يشتغل فوراً بعينتيك الموجودتين بالفعل:
     - B_shared_7000ms.wav (7 ثوانٍ + نصها المؤكد) لتوجيه Habibi-TTS المصري (S1)
     - B_new_10800ms.wav (10.8 ثانية) لتعزيز بصمة خامة عمر عبر OpenVoice V2 (S2)
     - ثم تمرير اختياري على سلسلة استوديو رِواق scripts/voice_finish.py (48kHz · -16 LUFS · -1.0 dBTP) (S3).
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import voice_finish
import transcribe_to_meta

# المسارات المؤقتة على قرص كولاب السريع (لا تستهلك شيئاً من 15GB الخاصة بـ Google Drive)
CACHE_DIR = Path(os.environ.get("RUWAQ_LOCAL_CACHE", "/content/ruwaq_local_cache"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))

DEFAULT_BENCHMARK_TEXT = "إزيّك، أنا عمر. خلّينا نحكي الحكاية بهدوء، واحدة واحدة."

_HABIBI_STATE = {"ema_model": None, "vocoder": None, "vocab_set": None, "ckpt_sha256": None}
_OPENVOICE_STATE = {"converter": None, "ckpt_sha256": None}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_deterministic_seed(seed: int = 42):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 1) محرك Habibi-TTS Specialized/EGY (Apache-2.0 · dialect_id=None) ────────
def load_habibi_egy(device: str | None = None):
    import torch
    from importlib.resources import files
    from huggingface_hub import hf_hub_download
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import load_model, load_vocoder

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _HABIBI_STATE["ema_model"] is not None:
        return _HABIBI_STATE["ema_model"], _HABIBI_STATE["vocoder"], _HABIBI_STATE["vocab_set"], device

    print(f"⏳ تحميل Habibi-TTS [Specialized/EGY · Apache-2.0] على {device} (في قرص كولاب المؤقت بدون استهلاك Drive)...")
    ckpt_path = hf_hub_download(
        repo_id="SWivid/Habibi-TTS",
        filename="Specialized/EGY/model_100000.safetensors",
        cache_dir=str(CACHE_DIR / "hf"),
    )
    vocab_path = hf_hub_download(
        repo_id="SWivid/Habibi-TTS",
        filename="Specialized/EGY/vocab.txt",
        cache_dir=str(CACHE_DIR / "hf"),
    )

    vocab_lines = Path(vocab_path).read_text(encoding="utf-8").splitlines()
    vocab_set = set(vocab_lines)

    cfg_path = str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml"))
    model_cfg = OmegaConf.load(cfg_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    ema_model = load_model(
        model_cls,
        model_arc,
        ckpt_path,
        mel_spec_type="vocos",
        vocab_file=vocab_path,
        device=device,
    )
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    _HABIBI_STATE.update({
        "ema_model": ema_model,
        "vocoder": vocoder,
        "vocab_set": vocab_set,
        "ckpt_path": ckpt_path,
        "vocab_path": vocab_path,
    })
    return ema_model, vocoder, vocab_set, device


def check_text_no_dialect_tags_and_oov(text: str, vocab_set: set[str]) -> dict:
    """حارس CA-02-D01: منع أي رموز لهجة Unified مثل ⑥〈…〉 وفحص الأحرف خارج المفردات."""
    cleaned = re.sub(r"⑥〈[^〉]*〉", "", text).strip()
    had_unified_tag = cleaned != text.strip()
    oov_chars = sorted({ch for ch in cleaned if not ch.isspace() and ch not in vocab_set})
    return {
        "clean_text": cleaned,
        "stripped_unified_tag": had_unified_tag,
        "oov_chars": oov_chars,
    }


def synthesize_s1_habibi_egy(
    gen_text: str,
    ref_audio_path: str,
    ref_text: str,
    out_s1_wav: str,
    seed: int = 42,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
) -> dict:
    """توليد S1 عبر Habibi-TTS Specialized/EGY مع dialect_id=None صريحاً (إصلاح CA-02-D01)."""
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    from habibi_tts.infer.utils_infer import infer_process

    ema_model, vocoder, vocab_set, device = load_habibi_egy()
    guard = check_text_no_dialect_tags_and_oov(gen_text, vocab_set)

    # إذا ترك المستخدم نص المرجع فارغاً، يفرغه preprocess_ref_audio_text تلقائياً أو نستخدم النص المؤكد
    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio_path, ref_text or "")

    set_deterministic_seed(seed)
    # ملاحظة حاسمة (إصلاح CA-02-D01): dialect_id=None إلزامي مع Specialized/EGY!
    wav_arr, sr, _ = infer_process(
        ref_audio_proc,
        ref_text_proc,
        guard["clean_text"],
        ema_model,
        vocoder,
        mel_spec_type="vocos",
        nfe_step=int(nfe_step),
        cfg_strength=float(cfg_strength),
        speed=float(speed),
        device=device,
        dialect_id=None,
    )

    wav_arr = np.asarray(wav_arr, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(out_s1_wav)), exist_ok=True)
    sf.write(out_s1_wav, wav_arr, int(sr), subtype="PCM_16")

    return {
        "s1_wav": out_s1_wav,
        "sr": int(sr),
        "duration_s": round(len(wav_arr) / sr, 3),
        "ref_text_used": ref_text_proc,
        "gen_text_used": guard["clean_text"],
        "dialect_id_used": None,
        "oov_chars": guard["oov_chars"],
        "seed": seed,
        "nfe_step": nfe_step,
    }


# ── 2) محرك OpenVoice V2 مع ترقيع التوافق الحديث لـ PyTorch 2.6+ و Librosa 0.11+ ──
def _patch_openvoice_modern_pytorch_librosa():
    """إصلاح أعطال OpenVoice المعروفة على PyTorch >= 2.2 (return_complex=False) و librosa >= 0.10."""
    import librosa
    import torch
    import openvoice.mel_processing as mp
    import openvoice.api as ov_api

    def _safe_spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
        dtype_device = f"{y.dtype}_{y.device}"
        key = f"{win_size}_{dtype_device}"
        if key not in mp.hann_window:
            mp.hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
        pad = int((n_fft - hop_size) / 2)
        y_pad = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec_c = torch.stft(
            y_pad,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=mp.hann_window[key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec_r = torch.view_as_real(spec_c)
        return torch.sqrt(spec_r.pow(2).sum(-1) + 1e-6)

    def _safe_mel_fn(sr, n_fft, n_mels, fmin, fmax):
        return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    mp.spectrogram_torch = _safe_spectrogram_torch
    mp.librosa_mel_fn = _safe_mel_fn
    ov_api.spectrogram_torch = _safe_spectrogram_torch


def load_openvoice_v2(device: str | None = None):
    import torch
    from huggingface_hub import hf_hub_download

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if _OPENVOICE_STATE["converter"] is not None:
        return _OPENVOICE_STATE["converter"], device

    _patch_openvoice_modern_pytorch_librosa()
    from openvoice.api import ToneColorConverter

    print(f"⏳ تحميل OpenVoice V2 [MIT License] على {device}...")
    cfg_path = hf_hub_download(
        repo_id="myshell-ai/OpenVoiceV2",
        filename="converter/config.json",
        cache_dir=str(CACHE_DIR / "hf"),
    )
    ckpt_path = hf_hub_download(
        repo_id="myshell-ai/OpenVoiceV2",
        filename="converter/checkpoint.pth",
        cache_dir=str(CACHE_DIR / "hf"),
    )

    converter = ToneColorConverter(cfg_path, device=device, enable_watermark=False)
    ckpt_dict = torch.load(ckpt_path, map_location=torch.device(device), weights_only=False)
    converter.model.load_state_dict(ckpt_dict["model"], strict=False)

    _OPENVOICE_STATE["converter"] = converter
    return converter, device


def convert_s2_openvoice(
    s1_wav_path: str,
    target_ref_wav_path: str,
    out_s2_wav: str,
    tau: float = 0.3,
) -> dict:
    """تحويل S1 إلى بصمة مرجع عمر (B_new_10800ms.wav) باستخدام استخراج SE الأصلي من الملفين."""
    converter, _ = load_openvoice_v2()
    src_se = converter.extract_se([s1_wav_path])
    tgt_se = converter.extract_se([target_ref_wav_path])
    os.makedirs(os.path.dirname(os.path.abspath(out_s2_wav)), exist_ok=True)
    converter.convert(
        audio_src_path=s1_wav_path,
        src_se=src_se,
        tgt_se=tgt_se,
        output_path=out_s2_wav,
        tau=float(tau),
    )
    data, sr = sf.read(out_s2_wav, dtype="float32")
    return {
        "s2_wav": out_s2_wav,
        "sr": int(sr),
        "duration_s": round(len(data) / sr, 3),
        "tau": float(tau),
    }


# ── 3) تشغيل اختبار المقارنة الثلاثي (S1 vs S2 vs S3 Riwaq Master) ─────────────
def run_ablation_test(
    gen_text: str,
    ref_shared_7s: str,
    ref_shared_transcript: str,
    ref_new_10s: str | None,
    seed: int = 42,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
    tau: float = 0.3,
    master_source_choice: str = "S1",
) -> tuple[str | None, str | None, str | None, str]:
    if not ref_shared_7s or not os.path.exists(ref_shared_7s):
        return None, None, None, "⚠️ يُرجى رفع ملف المرجع الأول (`B_shared_7000ms.wav` — 7 ثوانٍ) أولاً."
    if not (gen_text or "").strip():
        return None, None, None, "⚠️ يُرجى كتابة النص المراد توليده."

    out_dir = CACHE_DIR / "outputs_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_path = str(out_dir / "S1_Habibi_EGY_native.wav")
    s2_path = str(out_dir / "S2_Habibi_plus_OpenVoiceV2.wav")
    s3_path = str(out_dir / "S3_Riwaq_Studio_Master_48k.wav")

    s1_info = synthesize_s1_habibi_egy(
        gen_text=gen_text,
        ref_audio_path=ref_shared_7s,
        ref_text=(ref_shared_transcript or "").strip(),
        out_s1_wav=s1_path,
        seed=int(seed),
        nfe_step=int(nfe_step),
        cfg_strength=float(cfg_strength),
        speed=float(speed),
    )

    tgt_ref = ref_new_10s if (ref_new_10s and os.path.exists(ref_new_10s)) else ref_shared_7s
    s2_info = convert_s2_openvoice(
        s1_wav_path=s1_path,
        target_ref_wav_path=tgt_ref,
        out_s2_wav=s2_path,
        tau=float(tau),
    )

    chosen_for_master = s1_path if master_source_choice == "S1" else s2_path
    s3_info = voice_finish.finish(
        chosen_for_master,
        s3_path,
        target_lufs=-16.0,
        target_centroid=380.0,
        sr_out=48000,
        verbose=True,
    )

    oov_note = "لا يوجد (0 OOV ✅)" if not s1_info["oov_chars"] else f"⚠️ أحرف خارج المفردات: {s1_info['oov_chars']}"
    report = (
        f"### ✅ اكتمل التوليد والمقارنة في ثوانٍ (تراخيص تجارية نظيفة: Apache-2.0 + MIT · استهلاك Drive = 0 MB)\n"
        f"- **فحص `CA-02-D01`:** `dialect_id = None` (تم إثبات عدم حقن رموز Unified) · المفردات: {oov_note}\n"
        f"- **نص المرجع المعتمد لـ S1:** `{s1_info['ref_text_used']}`\n"
        f"- **`S1` (Habibi-TTS EGY الخام):** `{s1_info['duration_s']}s` @ `{s1_info['sr']}Hz`\n"
        f"- **`S2` (بعد محوّل OpenVoice V2 · `tau={tau}`):** `{s2_info['duration_s']}s` @ `{s2_info['sr']}Hz`\n"
        f"- **`S3` (ماستر استوديو رِواق من `{master_source_choice}`):** `48000Hz` · `LUFS {s3_info['post_lufs']}` · `Peak {s3_info['peak_dbtp']} dBTP` · `Centroid {s3_info['centroid_hz']}Hz`\n\n"
        f"👂 **قرار الاعتماد بأذنك أنت فقط:** اسمع الثلاثة بالترتيب واختار أنقاهم لهوية رِواق!"
    )
    return s1_path, s2_path, s3_path, report


# ── 4) توليد حزمة حلقة كاملة لنظام «رِواق» بنفس المحرك المختار ────────────────
def build_full_episode_zip_habibi(
    project_id: str,
    scenes_script: str,
    ref_shared_7s: str,
    ref_shared_transcript: str,
    ref_new_10s: str | None,
    use_openvoice_s2: bool = False,
    tau: float = 0.3,
    speed: float = 1.0,
    run_whisper: bool = True,
) -> tuple[str | None, str | None, str]:
    if not ref_shared_7s or not os.path.exists(ref_shared_7s):
        return None, None, "⚠️ ارفع ملف المرجع `B_shared_7000ms.wav` أولاً."

    pid = re.sub(r"[^A-Za-z0-9_\-]", "_", (project_id or "EP_001").strip())
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", (scenes_script or "").strip()) if b.strip()]
    if not blocks:
        return None, None, "⚠️ اكتب نص المشاهد أولاً."

    ep_root = CACHE_DIR / "episodes" / pid
    if ep_root.exists():
        shutil.rmtree(ep_root)
    audio_dir = ep_root / "public" / "projects" / pid / "audio"
    data_dir = ep_root / "data" / "projects" / pid
    outputs_dir = ep_root / "outputs"
    reports_dir = ep_root / "reports"
    work_dir = ep_root / "_work"
    for d in (audio_dir, data_dir, outputs_dir, reports_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)

    tgt_ref = ref_new_10s if (ref_new_10s and os.path.exists(ref_new_10s)) else ref_shared_7s
    vo_meta = []
    vo_whisper = []
    preview_parts = []
    cursor_s = 0.2

    for idx, blk in enumerate(blocks, start=1):
        first = blk.splitlines()[0].strip()
        parts = [p.strip() for p in first.split("|")]
        if len(parts) >= 3 and re.match(r"^[A-Za-z0-9_\-]+$", parts[0]):
            name = parts[0].replace(".wav", "")
            try:
                t_in = float(parts[1])
            except ValueError:
                t_in = cursor_s
            text = ("|".join(parts[2:]) + " " + " ".join(blk.splitlines()[1:])).strip()
        else:
            name = f"vo_{idx:02d}"
            t_in = cursor_s
            text = " ".join(x.strip() for x in blk.splitlines() if x.strip())

        s1_w = str(work_dir / f"{name}_s1.wav")
        s2_w = str(work_dir / f"{name}_s2.wav")
        fin_w = str(audio_dir / f"{name}.wav")

        synthesize_s1_habibi_egy(
            gen_text=text,
            ref_audio_path=ref_shared_7s,
            ref_text=(ref_shared_transcript or "").strip(),
            out_s1_wav=s1_w,
            seed=42,
            nfe_step=32,
            speed=speed,
        )
        if use_openvoice_s2:
            convert_s2_openvoice(s1_w, tgt_ref, s2_w, tau=tau)
            src_w = s2_w
        else:
            src_w = s1_w

        info = voice_finish.finish(src_w, fin_w, target_lufs=-16.0, target_centroid=380.0, sr_out=48000, verbose=True)
        vo_meta.append({"src": f"projects/{pid}/audio/{name}.wav", "t_in": round(t_in, 2), "volume_db": 0})
        vo_whisper.append({"path": fin_w, "t_in": round(t_in, 2)})
        cursor_s = round(t_in + info["duration_s"] + 0.55, 2)

        seg, _ = sf.read(fin_w, dtype="float32")
        preview_parts.append(seg)
        preview_parts.append(np.zeros(int(0.55 * 48000), dtype="float32"))

    preview_wav = str(ep_root / f"PREVIEW_{pid}.wav")
    sf.write(preview_wav, np.concatenate(preview_parts), 48000, subtype="PCM_16")

    # بناء episode_meta.json ونسخ الموسيقى من القالب الذهبي
    tmpl = REPO_ROOT / "data" / "projects" / "GOLDEN_001" / "episode_meta.json"
    meta = json.loads(tmpl.read_text(encoding="utf-8")) if tmpl.exists() else {"schema_version": "2.0", "project": {}}
    total_dur = max(15, int(round(cursor_s + 2.5)))
    meta["project"]["id"] = pid
    meta["audio"]["vo"] = vo_meta
    meta["edit"]["duration_s"] = total_dur
    meta["audio"]["music"] = {"src": f"projects/{pid}/audio/music.wav", "t_in": 0, "t_out": total_dur, "volume_db": -24}

    golden_aud = REPO_ROOT / "public" / "projects" / "GOLDEN_001" / "audio"
    for fn in ("music.wav", "stinger_in.wav", "stinger_out.wav"):
        if (golden_aud / fn).exists():
            shutil.copy2(golden_aud / fn, audio_dir / fn)

    meta_out = str(data_dir / "episode_meta.json")
    Path(meta_out).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if run_whisper:
        transcribe_to_meta.run_episode_transcription(vo_whisper, project_id=pid, out_dir=str(outputs_dir), meta_path=meta_out)

    shutil.rmtree(work_dir, ignore_errors=True)
    zip_out = str(CACHE_DIR / f"{pid}_riwaq_pack.zip")
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in ep_root.rglob("*"):
            if p.is_file() and p.name != os.path.basename(preview_wav):
                zf.write(p, p.relative_to(ep_root))

    return preview_wav, zip_out, f"✅ تم بناء حزمة الحلقة `{pid}` بالكامل وجاهزة للتحميل المباشر (`{zip_out}`) بدون استهلاك بايت واحد من Google Drive!"


def launch_habibi_gradio(share: bool = True):
    import gradio as gr

    with gr.Blocks(title="Ruwaq Habibi-EGY + OpenVoice V2 Studio (Zero Drive Storage)") as demo:
        gr.Markdown(
            "# 🎙️ مختبر «رِواق» للعامية المصرية (`Habibi-TTS Specialized/EGY` + `OpenVoice V2`)\n"
            "**مُصحَّح بالكامل من عيب `CA-02-D01` (`dialect_id=None`) · تراخيص تجارية معتمدة (`Apache-2.0` + `MIT`) · صفر استهلاك من مساحة Google Drive!**"
        )

        with gr.Row():
            ref7_in = gr.Audio(label="1️⃣ ارفع المقطع المرجعي الأول (B_shared_7000ms.wav — 7 ثوانٍ)", type="filepath")
            ref10_in = gr.Audio(label="2️⃣ ارفع المقطع المرجعي الثاني لـ OpenVoice (B_new_10800ms.wav — 10.8 ثانية · اختياري)", type="filepath")

        ref_txt_in = gr.Textbox(
            label="النص العربي الحرفي للمقطع الأول (B_shared_7000ms) — اتركه فارغاً فقط إذا أردت التفريغ التلقائي، ويُفضّل لصق النص المؤكد لدقة أعلى",
            lines=2,
            placeholder="اكتب هنا النص المنطوق في مقطع الـ 7 ثوانٍ بالضبط...",
        )

        with gr.Tabs():
            with gr.TabItem("🔬 1) اختبار استعادة الهوية (CA-02: S1 vs S2 vs S3 Master)"):
                gen_txt_in = gr.Textbox(
                    label="النص المصري الجديد المراد نطقه بصوت عمر",
                    lines=3,
                    value=DEFAULT_BENCHMARK_TEXT,
                )
                with gr.Row():
                    seed_in = gr.Number(value=42, precision=0, label="Seed")
                    nfe_in = gr.Slider(16, 64, value=32, step=4, label="خطوات الجودة NFE (32 قياسي)")
                    cfg_in = gr.Slider(1.0, 3.5, value=2.0, step=0.2, label="قوة التوجيه CFG")
                    spd_in = gr.Slider(0.8, 1.2, value=1.0, step=0.05, label="السرعة Speed")
                    tau_in = gr.Slider(0.1, 0.8, value=0.3, step=0.05, label="قوة تحويل OpenVoice (tau)")
                    m_choice = gr.Radio(["S1", "S2"], value="S1", label="مصدر الماستر النهائي S3")

                btn_test = gr.Button("🚀 شغّل الاختبار المزدوج (S1 + S2 + S3 Master) في ثوانٍ", variant="primary")
                with gr.Row():
                    s1_out = gr.Audio(label="S1: Habibi-TTS EGY الأصلي (24kHz)", type="filepath")
                    s2_out = gr.Audio(label="S2: بعد تحويل OpenVoice V2", type="filepath")
                    s3_out = gr.Audio(label="S3: بعد سلسلة استوديو رِواق (48kHz / -16 LUFS)", type="filepath")
                rep_md = gr.Markdown()

                btn_test.click(
                    fn=run_ablation_test,
                    inputs=[gen_txt_in, ref7_in, ref_txt_in, ref10_in, seed_in, nfe_in, cfg_in, spd_in, tau_in, m_choice],
                    outputs=[s1_out, s2_out, s3_out, rep_md],
                )

            with gr.TabItem("🎬 2) إنتاج حزمة حلقة «رِواق» كاملة (بدون Google Drive)"):
                with gr.Row():
                    ep_id = gr.Textbox(value="EP_001", label="معرّف الحلقة (Project ID)")
                    use_ov = gr.Checkbox(value=False, label="تمرير على OpenVoice V2 قبل الماسترينج (فعّلها إذا أعجبك S2 أكثر من S1)")
                    use_wh = gr.Checkbox(value=True, label="توليد كابشنز Faster-Whisper وتحديث episode_meta.json")
                ep_script = gr.Textbox(
                    label="سكريبت مشاهد الحلقة (افصل كل مشهد بسطر فارغ)",
                    lines=7,
                    value="في زاوية بعيدة من ذاكرة المدن، حكاية محدش فكر يفتح دفاترها من سبعمئة سنة.\n\nأكتر من 73% من المخطوطات القديمة لسه متقريتش، بس كل سطر فيها بيحكي سر جديد.",
                )
                btn_ep = gr.Button("📦 توليد حزمة الحلقة الكاملة (<ID>_riwaq_pack.zip)", variant="primary")
                ep_prev = gr.Audio(label="معاينة صوت الحلقة الماستر", type="filepath")
                ep_zip = gr.File(label="تحميل ملف ZIP الجاهز لمستودع رِواق")
                ep_msg = gr.Markdown()

                btn_ep.click(
                    fn=lambda pid, scr, r7, rt, r10, uov, tau_v, spd_v, uwh: build_full_episode_zip_habibi(
                        project_id=pid,
                        scenes_script=scr,
                        ref_shared_7s=r7,
                        ref_shared_transcript=rt,
                        ref_new_10s=r10,
                        use_openvoice_s2=uov,
                        tau=tau_v,
                        speed=spd_v,
                        run_whisper=uwh,
                    ),
                    inputs=[ep_id, ep_script, ref7_in, ref_txt_in, ref10_in, use_ov, tau_in, spd_in, use_wh],
                    outputs=[ep_prev, ep_zip, ep_msg],
                )

    demo.launch(share=share, inline=True, debug=False)
    return demo
