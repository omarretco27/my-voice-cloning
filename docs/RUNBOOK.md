# RUNBOOK — تشغيل النظام من الصفر للتسليم

> **الطريقة المعتمدة للريندر: GitHub (سحابي)** — راجع `docs/GITHUB_SETUP.md`.
> الأوامر المحلية هنا للمعاينة السريعة فقط (جهاز 8GB لا يُنصح بريندر 1080p عليه).

---

## 0) تجهيز أول مرة
```bash
npm install
# الإعدادات موجودة مسبقًا في data/ (brand.json · voice_profiles.json)
```

## 1) المعاينة التفاعلية
```bash
npm run studio        # يفتح Remotion Studio
```

## 2) الريندر السحابي (الطريقة الرسمية) ☁️
1. ارفع المستودع على GitHub مرة واحدة (دليل كامل: `docs/GITHUB_SETUP.md`).
2. تبويب **Actions** → **render-episode** → **Run workflow**:
   - `project_id` = معرّف المشروع (مثال `GOLDEN_001`)
   - `draft` = «نعم» لمسوّدة نصف دقة (أسرع 4×) أو «لا» للنسخة الكاملة
3. نزّل النتائج من **Artifacts**: فيديو MP4 + كابشنز AR/EN + تقرير QC.

## 3) الريندر المحلي (معاينة/مسوّدات فقط)
```bash
npm run render:golden:draft     # نصف دقة — الأسرع على جهازك
npm run render:golden           # كامل 1080p (تجنّبه إن أمكن — استخدم GitHub)
npm run still                   # إطار مفرد للفحص البصري السريع
```

## 4) الصوت — سلسلة الاستوديو (إلزامية لكل تعليق صوتي)
```bash
python scripts/voice_finish.py public/projects/D04/audio/vo_raw.wav
# المخرج: LUFS −16 · ذروة ≤ −1 dBTP · بصمة مطابقة (بدون أي إزاحة نغمة — قاعدة مقفولة)
```
> ⛔ **ممنوع pitch-shift على التعليق الصوتي** — المطابقة الطبيعية تحدث في مرحلة التوليد (الصوت المستنسخ الرسمي).

## 5) الكابشنز
```bash
python scripts/make_srt.py data/projects/D04/episode_meta.json outputs
# CAPTIONS_AR_D04.srt + CAPTIONS_EN_D04.srt — مع فرض: سطران · 28 حرفًا · عزل BiDi
```

## 6) بناء ملف الريندر (Props)
```bash
python scripts/make_props.py data/projects/D04      # يولّد render_props.json {meta, brand}
```

## 7) بوابة الجودة (قبل أي تسليم)
```bash
python scripts/qc.py data/projects/D04/episode_meta.json exports reports
# reports/QC_D04.md — أي ❌ = لا تسليم
```

## 8) فيديو جديد — الخطوات الكاملة
1. `copy data\projects\GOLDEN_001 data\projects\D04` (ويندوز) أو `cp -r` (لينكس/ماك).
2. عدّل `data/projects/D04/episode_meta.json`:
   - `project.id` = `D04` · العنوان · النوع · المنصات
   - `story.scenes` = المشاهد بتوقيتاتها
   - `captions.data` = الكابشنز (كلمة مفتاحية واحدة كحد أقصى)
   - `audio.*` = مسارات الأصوات داخل `public/projects/D04/audio/`
3. ضع الأصوات في `public/projects/D04/audio/`.
4. Commit + Push → Actions → Run workflow → `D04` → نزّل النتيجة.

## 9) خط الإنتاج للصوت (عند توفر الصوت الرسمي)
1. ولّد التعليق بالصوت المستنسخ من ElevenLabs (نفس إعدادات البصمة المقفولة).
2. طبّق `voice_finish.py` فقط (بلا إزاحة نغمة).
3. ضع الملفات في `public/projects/<ID>/audio/` وحدّث `episode_meta` — **لا شيء آخر يتغير**.

## 10) قائمة ما قبل التسليم (10 ثوانٍ)
1. `reports/QC_<ID>.md` بلا ❌
2. `reports/SOURCES_<ID>.txt` محدّث (فارغ مسموح للمحتوى الأصلي)
3. التسمية مطابقة (`docs/NAMING` داخل حزمة القوالب)
4. لا وسم `DRAFT_VO` على أي ملف مخصص للنشر
5. نسخة أرشيفية على الـHDD

## 11) حل المشاكل (سريع)
| العرض | السبب | الحل |
|---|---|---|
| أحرف عربية مقطّعة في الريندر | الخطوط لم تُحمَّل | تأكد من `public/fonts` كاملة (5 ملفات TTF) — التحميل تلقائي من `src/engine/fonts.ts` |
| فشل الريندر في CI عند المتصفح | Chromium | أعد تشغيل الـworkflow — الخطوة `npx remotion browser ensure` تعالجها |
| «لم يتم العثور على props» | ملف render_props ناقص | شغّل `make_props.py` (الـCI يفعلها تلقائيًا) |
| فيديو صامت | مسارات الصوت | راجع `audio.vo[].src` مطابقة لملفات `public/projects/<ID>/audio/` |
| الريندر بطيء محليًا | ذاكرة 8GB | استخدم `draft` في السحابة أو `render:golden:draft` محليًا |
