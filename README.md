# رِواق — نظام إنتاج الفيديو واستنساخ الصوت (Riwaq Production & Voice Studio)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omarretco27/my-voice-cloning/blob/arena/a44deae5-my-voice-cloning/colab/Riwaq_Voice_Cloning_Studio.ipynb)

> **محرّك واحد، هويات متعددة + استوديو صوتي سحابي مجاني.**
> فيديو جديد = سكريبت في نوتبوك كولاب ⬅️ حزمة صوت وكابشنز جاهزة (`ZIP`) ⬅️ ريندر سحابي على GitHub Actions. صفر تعديل في الكود.

## الحالة الحالية
| المرحلة | الحالة |
|---|---|
| نظام التصميم والهوية (`data/brand.json`) | ✅ جاهز |
| استوديو استنساخ الصوت المجاني (`colab/Riwaq_Voice_Cloning_Studio.ipynb`) | ✅ جاهز (Text➔Omar + Voice➔Omar عبر RVC v2 48kHz + XTTS-v2) |
| مدقق ومُقطّع التسجيلات الخام (`scripts/prepare_dataset.py`) | ✅ جاهز (فحص SNR/F0 + عزل لطيف + تقطيع على الوقفات + Golden Ref) |
| تفريغ الكابشنز كلمة-بكلمة (`scripts/transcribe_to_meta.py`) | ✅ جاهز (Faster-Whisper ➔ `episode_meta.json` + `SRT` ≤28 حرفاً) |
| سلسلة استوديو الصوت (`scripts/voice_finish.py` v2.0) | ✅ جاهز (`48kHz` · `LUFS −16` · `TP −1.0 dBTP` · مركز طيف `380Hz`) |
| الريندر السحابي (`render-episode` في `.github/workflows/`) | ✅ مُفعَّل وجاهز |
| الاختبار الذهبي `GOLDEN_001` وبوابة الجودة (`scripts/qc.py`) | ✅ يمر بجميع الفحوص |

---

## 🎙️ 1) استوديو استنساخ الصوت وإنتاج الحلقات (Google Colab المجاني)
تخلّص بالكامل من الاشتراكات المدفوعة وأنتج تعليقك الصوتي الرسمي (`omar_el_clone_v1`) مجاناً على كارت **T4 GPU**:
- **الدليل الكامل لتجميع التسجيلات والرفع على Google Drive:** [`docs/VOICE_CLONING_GUIDE_AR.md`](docs/VOICE_CLONING_GUIDE_AR.md)
- **سكريبت المعايرة الصوتية بالعامية المصرية (اختياري لتزويد الداتا):** [`docs/OMAR_CALIBRATION_SCRIPT_AR_EG.md`](docs/OMAR_CALIBRATION_SCRIPT_AR_EG.md)
- **النوتبوك الشامل بواجهة عربية تفاعلية (Gradio):** [`colab/Riwaq_Voice_Cloning_Studio.ipynb`](colab/Riwaq_Voice_Cloning_Studio.ipynb)

---

## ☁️ 2) الريندر السحابي للفيديو (GitHub Actions — جهازك لا يعمل)
1. فك ضغط ملف الحلقة الناتج من كولاب (`<ID>_riwaq_pack.zip`) في جذر المستودع واعمل Commit + Push.
2. ادخل تبويب **Actions** ⬅️ **render-episode** ⬅️ **Run workflow** ⬅️ اكتب معرّف المشروع (مثلاً `GOLDEN_001` أو `EP_001`).
3. نزّل النتائج من **Artifacts**: فيديو `MP4` + كابشنز `AR/EN SRT` + تقرير الجودة `QC.md`.

---

## البنية المعمارية للمستودع
```text
├── colab/
│   ├── Riwaq_Voice_Cloning_Studio.ipynb   ← دفتر Google Colab الشامل (تدريب + واجهة إنتاج الحلقات)
│   └── riwaq_colab_app.py                 ← محرك استنساخ الصوت (RVC v2 + XTTS + ar-EG) وتغليف الحلقات
├── scripts/
│   ├── prepare_dataset.py                 ← فحص وتنظيف وتقطيع التسجيلات الخام 48kHz واستخراج Golden Ref
│   ├── voice_finish.py                    ← سلسلة استوديو الماسترينج (48kHz · LUFS -16 · TP -1dB · 380Hz)
│   ├── transcribe_to_meta.py              ← تفريغ Faster-Whisper وبناء كابشنز episode_meta.json تلقائياً
│   ├── make_srt.py                        ← توليد ملفات SRT ثنائية اللغة مع عزل BiDi
│   ├── make_props.py                      ← بناء render_props.json لمحرك Remotion
│   └── qc.py                              ← بوابة فحص الجودة الآلية قبل التسليم
├── data/
│   ├── brand.json · voice_profiles.json   ← دستور الهوية البصرية والصوتية لـ «رِواق»
│   └── projects/GOLDEN_001/episode_meta.json
├── src/                                   ← محرك الفيديو ومكوّنات Remotion السبعة (A–G + Captions)
├── public/                                ← الخطوط المعتمدة (OFL) وأصول الحلقات الصوتية
├── .github/workflows/render-episode.yml   ← خط الريندر السحابي التلقائي
└── docs/                                  ← أدلة التشغيل واستنساخ الصوت والرفع على GitHub
```

## القواعد الإلزامية (من دستور الهوية)
- **الألوان:** `#1C1B19 #242218 #F3EBDB #EDE6D6 #A98A4B #C2A469 #EFE6D2 #5E2230 #8A7F6B #B9AE94` فقط.
- **الخطوط:** **Amiri** للعناوين · **IBM Plex Sans Arabic** للكابشنز والمعلومات.
- **الكابشنز:** سطران أقصى · `≤28` حرفاً عربياً للسطر · إبراز الكلمات المفتاحية والأرقام بالذهبي · مدة `≥1.2s`.
- **الصوت:** `WAV 48kHz` · ارتفاع `LUFS −16 ±0.5` · ذروة `≤ −1.0 dBTP` · **ممنوع `pitch-shift` في المونتاج** (البصمة تأتي طبيعية من موديل الاستنساخ).
