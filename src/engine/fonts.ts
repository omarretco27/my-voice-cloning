import { loadFont } from "@remotion/fonts";
import { continueRender, delayRender, staticFile } from "remotion";

/**
 * تحميل الخطوط الرسمية قبل أي ريندر — بدونه الكابشنز العربية تخرج بأحرف مقطّعة.
 * الخطوط بترخيص OFL داخل public/fonts.
 */
const FONTS: { family: string; file: string; weight: string }[] = [
  { family: "Amiri", file: "fonts/Amiri-Bold.ttf", weight: "700" },
  { family: "Amiri", file: "fonts/Amiri-Regular.ttf", weight: "400" },
  { family: "IBM Plex Sans Arabic", file: "fonts/IBMPlexSansArabic-Bold.ttf", weight: "700" },
  { family: "IBM Plex Sans Arabic", file: "fonts/IBMPlexSansArabic-Medium.ttf", weight: "500" },
  { family: "IBM Plex Sans Arabic", file: "fonts/IBMPlexSansArabic-Regular.ttf", weight: "400" },
];

const handle = delayRender("Riwaq: تحميل الخطوط الرسمية");

Promise.all(
  FONTS.map((f) =>
    loadFont({ family: f.family, url: staticFile(f.file), weight: f.weight })
  )
)
  .then(() => continueRender(handle))
  .catch(() => continueRender(handle));
