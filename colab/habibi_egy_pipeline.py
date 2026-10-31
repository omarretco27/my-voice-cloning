#!/usr/bin/env python3
"""habibi_egy_pipeline.py — استوديو «رِواق» للعامية المصرية (Habibi-TTS Specialized/EGY + OpenVoice V2) v2.1

متوافق 100% مع Google Colab 2026 (Python 3.13.15 + Ubuntu 24.04 + TorchAudio 2.9+):
  1) ترقيع audioop تلقائياً لـ Python 3.13 (PEP 594) قبل استيراد f5_tts / pydub.
  2) ترقيع torchaudio.load و torchaudio.save عبر soundfile لتفادي عطل TorchCodec.
  3) تمرير مصفوفة صوت 16kHz مباشرة إلى Faster-Whisper لتفادي عطل PyAV (av.open) على Python 3.13.
  4) القص الذكي التلقائي للمقطع المرجعي (≤ 9.2 ثانية) لحماية معادلة (22 - duration) في Habibi-TTS.
  5) إصلاح عيب ToneColorConverter.__init__(enable_watermark) في OpenVoice V2 + دعم دمج كل مقاطع عمر النقية لبصمة أقوى.
  6) إصلاح CA-02-D01 رسمياً: dialect_id=None مع أوزان Specialized/EGY (Apache-2.0 التجاري).
"""
from __future__ import annotations

import gc
import hashlib
import os
import re
import sys
import traceback
import types
from pathlib import Path
import numpy as np
import soundfile as sf

# ── 0) حماية Python 3.13 (PEP 594): ضمان وجود audioop قبل استيراد pydub داخل f5_tts ──
try:
    import audioop  # noqa: F401
except ImportError:
    try:
        import pyaudioop as _pyaudioop
        sys.modules["audioop"] = _pyaudioop
    except ImportError:
        sys.modules["audioop"] = types.ModuleType("audioop")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

for extra_src in ("/content/Habibi-TTS/src", "/content/OpenVoice"):
    if os.path.exists(extra_src) and extra_src not in sys.path:
        sys.path.insert(0, extra_src)

import voice_finish
import prepare_dataset
import transcribe_to_meta

CACHE_DIR = Path(os.environ.get("RUWAQ_LOCAL_CACHE", "/content/ruwaq_local_cache"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))

DEFAULT_BENCHMARK_TEXT = "إزيّك، أنا عمر. خلّينا نحكي الحكاية بهدوء، واحدة واحدة."

_HABIBI_STATE = {"ema_model": None, "vocoder": None, "vocab_set": None}
_OPENVOICE_STATE = {"converter": None, "tgt_se_cache": {}}
_WHISPER_REF_CACHE: dict[str, tuple[str, str]] = {}


def patch_torchaudio_soundfile():
    import torch
    import torchaudio

    # تعطيل cuDNN الفرعي لتفادي خطأ CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH على Google Colab CUDA 13
    # مع الإبقاء الكامل على تسريع كرت الشاشة T4 GPU عبر نوى CUDA الأصيلة في PyTorch!
    torch.backends.cudnn.enabled = False
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    if getattr(torchaudio, "_ruwaq_sf_patched", False):
        return

    def _safe_ta_load(filepath, *args, **kwargs):
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        tensor = torch.from_numpy(np.ascontiguousarray(data.T))
        return tensor, int(sr)

    def _safe_ta_save(filepath, src, sample_rate, *args, **kwargs):
        arr = src.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T
        sf.write(str(filepath), arr, int(sample_rate), subtype="PCM_16")

    torchaudio.load = _safe_ta_load
    torchaudio.save = _safe_ta_save
    torchaudio._ruwaq_sf_patched = True


def set_deterministic_seed(seed: int = 42):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_safe_habibi_reference(ref_audio_path: str, ref_text: str = "") -> tuple[str, str]:
    """يضمن ألا يتجاوز المرجع 9.2 ثانية ويفرّغ نصه بدقة عبر Faster-Whisper بدون PyAV."""
    cache_key = f"{os.path.abspath(ref_audio_path)}::{os.path.getsize(ref_audio_path)}::{(ref_text or '').strip()}"
    if cache_key in _WHISPER_REF_CACHE:
        return _WHISPER_REF_CACHE[cache_key]

    x = prepare_dataset.load_audio_mono_48k(ref_audio_path, 24000)
    sr = 24000
    dur = len(x) / sr

    safe_dir = CACHE_DIR / "prepared_refs"
    safe_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
    safe_wav = str(safe_dir / f"habibi_ref_{digest}.wav")

    was_trimmed = False
    if dur > 10.2:
        chunks = prepare_dataset.slice_on_pauses(x, sr=sr, min_sec=5.5, max_sec=9.2)
        if chunks:
            x = max(chunks, key=lambda c: float(np.sqrt(np.mean(c ** 2))))
        else:
            x = x[: int(9.0 * sr)]
        was_trimmed = True
        print(f"✂ تم قص المرجع تلقائياً من {dur:.1f}s إلى {len(x)/sr:.2f}s ليناسب نافذة Habibi-TTS القياسية.")

    mx = float(np.max(np.abs(x)))
    if mx > 0.005:
        x = (x / mx * 0.85).astype(np.float32)
    sf.write(safe_wav, x, sr, subtype="PCM_16")

    final_ref_text = (ref_text or "").strip()
    if not final_ref_text or was_trimmed:
        print("📝 جاري استخراج النص الحرفي للمقطع المرجعي عبر Faster-Whisper لضمان مطابقة المخارج...")
        _, transcribed = transcribe_to_meta.transcribe_file(safe_wav, model_size="auto")
        final_ref_text = transcribed.strip()

    if not final_ref_text:
        final_ref_text = "بسم الله الرحمن الرحيم، أهلاً بيكم في رواق."

    final_ref_text = final_ref_text.strip() + "  "
    _WHISPER_REF_CACHE[cache_key] = (safe_wav, final_ref_text)
    return safe_wav, final_ref_text


def load_habibi_egy(device: str | None = None):
    patch_torchaudio_soundfile()
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

    print(f"⏳ تحميل Habibi-TTS [Specialized/EGY · Apache-2.0] على {device}...")
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

    vocab_set = set(Path(vocab_path).read_text(encoding="utf-8").splitlines())
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
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device, hf_cache_dir=str(CACHE_DIR / "hf"))

    _HABIBI_STATE.update({
        "ema_model": ema_model,
        "vocoder": vocoder,
        "vocab_set": vocab_set,
    })
    return ema_model, vocoder, vocab_set, device


def check_text_no_dialect_tags_and_oov(text: str, vocab_set: set[str]) -> dict:
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
    patch_torchaudio_soundfile()
    safe_ref_wav, safe_ref_text = prepare_safe_habibi_reference(ref_audio_path, ref_text)
    ema_model, vocoder, vocab_set, device = load_habibi_egy()

    from habibi_tts.infer.utils_infer import infer_process

    guard = check_text_no_dialect_tags_and_oov(gen_text, vocab_set)
    set_deterministic_seed(seed)

    wav_arr, sr, _ = infer_process(
        safe_ref_wav,
        safe_ref_text,
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

    if wav_arr is None or len(wav_arr) == 0:
        raise RuntimeError("لم ينتج محرك Habibi-TTS عينات صوتية؛ تحقق من النص المدخل.")

    wav_arr = np.asarray(wav_arr, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(out_s1_wav)), exist_ok=True)
    sf.write(out_s1_wav, wav_arr, int(sr), subtype="PCM_16")

    return {
        "s1_wav": out_s1_wav,
        "sr": int(sr),
        "duration_s": round(len(wav_arr) / sr, 3),
        "ref_text_used": safe_ref_text.strip(),
        "gen_text_used": guard["clean_text"],
        "dialect_id_used": None,
        "oov_chars": guard["oov_chars"],
        "seed": seed,
        "nfe_step": nfe_step,
    }


def _patch_openvoice_modern_pytorch_librosa():
    import librosa
    import torch
    import openvoice.mel_processing as mp
    import openvoice.api as ov_api

    if getattr(ov_api, "_ruwaq_ov_patched", False):
        return

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

    # إصلاح عيب upstream في ToneColorConverter.__init__ عند تمرير enable_watermark=False
    def _safe_tcc_init(self, *args, **kwargs):
        enable_wm = kwargs.pop("enable_watermark", False)
        ov_api.OpenVoiceBaseClass.__init__(self, *args, **kwargs)
        if enable_wm:
            import wavmark
            self.watermark_model = wavmark.load_model().to(self.device)
        else:
            self.watermark_model = None
        self.version = getattr(self.hps, "_version_", "v2")

    # استبدال librosa.load داخل OpenVoice بقارئ soundfile/scipy المباشر لتفادي audioread على Python 3.13
    def _safe_extract_se(self, ref_wav_list, se_save_path=None):
        if isinstance(ref_wav_list, (str, Path)):
            ref_wav_list = [str(ref_wav_list)]
        device = self.device
        hps = self.hps
        gs = []
        for fname in ref_wav_list:
            audio_ref = prepare_dataset.load_audio_mono_48k(str(fname), int(hps.data.sampling_rate))
            y = torch.from_numpy(audio_ref).float().to(device).unsqueeze(0)
            y = _safe_spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            ).to(device)
            with torch.no_grad():
                g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                gs.append(g.detach())
        gs = torch.stack(gs).mean(0)
        if se_save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(se_save_path)), exist_ok=True)
            torch.save(gs.cpu(), se_save_path)
        return gs

    def _safe_convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3, message="default"):
        hps = self.hps
        audio = prepare_dataset.load_audio_mono_48k(str(audio_src_path), int(hps.data.sampling_rate))
        with torch.no_grad():
            y = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
            spec = _safe_spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            ).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            out_audio = (
                self.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau)[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
            if output_path is None:
                return out_audio
            sf.write(str(output_path), out_audio, int(hps.data.sampling_rate), subtype="PCM_16")

    mp.spectrogram_torch = _safe_spectrogram_torch
    mp.librosa_mel_fn = _safe_mel_fn
    ov_api.spectrogram_torch = _safe_spectrogram_torch
    ov_api.ToneColorConverter.__init__ = _safe_tcc_init
    ov_api.ToneColorConverter.extract_se = _safe_extract_se
    ov_api.ToneColorConverter.convert = _safe_convert
    ov_api._ruwaq_ov_patched = True


def load_openvoice_v2(device: str | None = None):
    patch_torchaudio_soundfile()
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
    target_ref_wav_path: str | list[str],
    out_s2_wav: str,
    tau: float = 0.3,
) -> dict:
    converter, _ = load_openvoice_v2()
    src_se = converter.extract_se([s1_wav_path])

    if isinstance(target_ref_wav_path, (str, Path)):
        ref_list = [str(target_ref_wav_path)]
    else:
        ref_list = [str(p) for p in target_ref_wav_path]

    cache_key = "::".join(sorted(ref_list))
    if cache_key not in _OPENVOICE_STATE["tgt_se_cache"]:
        _OPENVOICE_STATE["tgt_se_cache"][cache_key] = converter.extract_se(ref_list)
    tgt_se = _OPENVOICE_STATE["tgt_se_cache"][cache_key]

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
        "segments_averaged": len(ref_list),
    }


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
    try:
        if not ref_shared_7s or not os.path.exists(ref_shared_7s):
            return None, None, None, "⚠️ يُرجى رفع ملف صوتي مرجعي أولاً في المربع رقم (1) بالأعلى قبل الضغط على زر التشغيل!"
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
            f"### ✅ اكتمل التوليد والمقارنة بنجاح!\n"
            f"- **فحص `CA-02-D01`:** `dialect_id = None` · المفردات: {oov_note}\n"
            f"- **نص المرجع المستخدم:** `{s1_info['ref_text_used']}`\n"
            f"- **`S1` (Habibi-TTS EGY الأصلي):** `{s1_info['duration_s']}s` @ `{s1_info['sr']}Hz`\n"
            f"- **`S2` (بعد محوّل OpenVoice V2):** `{s2_info['duration_s']}s` @ `{s2_info['sr']}Hz`\n"
            f"- **`S3` (ماستر استوديو رِواق):** `48000Hz` · `LUFS {s3_info['post_lufs']}` · `Peak {s3_info['peak_dbtp']} dBTP` · `Centroid {s3_info['centroid_hz']}Hz`"
        )
        return s1_path, s2_path, s3_path, report
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return None, None, None, f"### ❌ حدث خطأ تقني — التفاصيل الكاملة للتشخيص:\n```text\n{exc}\n\n{tb}\n```"


def launch_habibi_gradio(share: bool = True):
    import gradio as gr

    with gr.Blocks(title="Ruwaq Habibi-EGY + OpenVoice V2 Studio v2.1") as demo:
        gr.Markdown("# 🎙️ مختبر «رِواق» للعامية المصرية (`Habibi-TTS Specialized/EGY` + `OpenVoice V2`) — v2.1")
        with gr.Row():
            ref7_in = gr.Audio(label="1️⃣ ارفع ملف صوتي بصوتك", type="filepath")
            ref10_in = gr.Audio(label="2️⃣ ملف مرجعي ثانٍ لـ OpenVoice (اختياري)", type="filepath")
        ref_txt_in = gr.Textbox(label="النص العربي للمرجع (اختياري)", lines=2)
        gen_txt_in = gr.Textbox(label="النص المصري المراد نطقه بصوت عمر", lines=3, value=DEFAULT_BENCHMARK_TEXT)
        with gr.Row():
            seed_in = gr.Number(value=42, precision=0, label="Seed")
            nfe_in = gr.Slider(16, 64, value=32, step=4, label="خطوات الجودة NFE")
            cfg_in = gr.Slider(1.0, 3.5, value=2.0, step=0.2, label="CFG")
            spd_in = gr.Slider(0.8, 1.2, value=1.0, step=0.05, label="Speed")
            tau_in = gr.Slider(0.1, 0.8, value=0.3, step=0.05, label="OpenVoice tau")
            m_choice = gr.Radio(["S1", "S2"], value="S1", label="مصدر الماستر S3")
        btn_test = gr.Button("🚀 شغّل الاختبار المزدوج (S1 + S2 + S3 Master)", variant="primary")
        with gr.Row():
            s1_out = gr.Audio(label="S1: Habibi-TTS EGY الأصلي", type="filepath")
            s2_out = gr.Audio(label="S2: بعد OpenVoice V2", type="filepath")
            s3_out = gr.Audio(label="S3: ماستر استوديو رِواق (48kHz / -16 LUFS)", type="filepath")
        rep_md = gr.Markdown()
        btn_test.click(
            fn=run_ablation_test,
            inputs=[gen_txt_in, ref7_in, ref_txt_in, ref10_in, seed_in, nfe_in, cfg_in, spd_in, tau_in, m_choice],
            outputs=[s1_out, s2_out, s3_out, rep_md],
        )
    demo.launch(share=share, inline=True, debug=True)
    return demo
