import React from "react";
import { AbsoluteFill } from "remotion";
import { useTheme, scalePx } from "../engine/theme";
import type { CaptionLine } from "../engine/types";

/**
 * كابشنز بثلاثة أنماط:
 *  - clean_bottom      : نظيف أسفل الشاشة
 *  - keyword_highlight : إبراز الكلمات المؤثرة والأرقام بالذهبي (النمط الافتراضي)
 *  - bilingual_stack   : عربي فوق (Plex Bold) + إنجليزي تحته (Plex Regular 60%)
 * الموضع دائمًا داخل نطاق الكابشنز الآمن من الثيم.
 */
export const Captions: React.FC<{
  line: CaptionLine;
  style: "clean_bottom" | "keyword_highlight" | "bilingual_stack";
}> = ({ line, style }) => {
  const { color, bodyFont, safe, width } = useTheme();
  const [y0, y1] = safe.captionY;
  const fontSize = scalePx(56, width);
  const enSize = fontSize * 0.6;

  const renderAr = () => {
    if (style !== "keyword_highlight" || !line.keywords?.length) return line.ar;
    const parts: React.ReactNode[] = [];
    let rest = line.ar;
    line.keywords.forEach((kw, i) => {
      const idx = rest.indexOf(kw);
      if (idx === -1) return;
      parts.push(rest.slice(0, idx));
      parts.push(
        <span key={i} style={{ color: color("gold_light") }}>
          {kw}
        </span>
      );
      rest = rest.slice(idx + kw.length);
    });
    parts.push(rest);
    return parts;
  };

  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "flex-start" }}>
      <div
        style={{
          position: "absolute",
          top: (y0 + y1) / 2,
          transform: "translateY(-50%)",
          maxWidth: width - safe.side * 2,
          textAlign: "center",
        }}
      >
        <div
          style={{
            display: "inline-block",
            background: "#0f0e0ce8",
            border: `1px solid ${color("gold")}33`,
            borderRadius: width * 0.009,
            padding: `${width * 0.012}px ${width * 0.02}px`,
          }}
        >
          <div
            style={{
              fontFamily: bodyFont,
              fontWeight: 700,
              fontSize,
              color: color("parchment"),
              lineHeight: 1.42,
            }}
          >
            {renderAr()}
          </div>
          {style === "bilingual_stack" && line.en ? (
            <div
              style={{
                fontFamily: bodyFont,
                fontWeight: 400,
                fontSize: enSize,
                color: color("stone_light"),
                marginTop: width * 0.006,
                direction: "ltr",
              }}
            >
              {line.en}
            </div>
          ) : null}
        </div>
      </div>
    </AbsoluteFill>
  );
};
