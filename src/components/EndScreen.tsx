import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { useTheme, scalePx } from "../engine/theme";
import { textTiming } from "../engine/timing";
import { BrandMark } from "./BrandMark";

/** G — شاشة النهاية: مطابقة للأصل الرسمي + مساحات NEXT STORY / SUBSCRIBE / FOLLOW */
export const EndScreen: React.FC<{ tease?: string; durFrames: number }> = ({ tease, durFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { color, headlineFont, bodyFont, width } = useTheme();
  const t = textTiming(frame, fps, 14, durFrames - 10, width * 0.01);

  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: width * 0.03 }}>
      <div style={{ opacity: t.opacity, transform: `translateY(${t.translateY}px)`, textAlign: "center" }}>
        <div style={{ display: "flex", justifyContent: "center" }}>
          <BrandMark sizePx={width * 0.085} color={color("gold_light")} />
        </div>
        <div
          style={{
            fontFamily: headlineFont,
            fontWeight: 700,
            fontSize: scalePx(64, width),
            color: color("parchment"),
            marginTop: width * 0.024,
          }}
        >
          حكايةٌ أخرى تنتظر أن تُروى
        </div>
        {tease ? (
          <div
            style={{
              fontFamily: bodyFont,
              fontSize: scalePx(30, width),
              color: color("stone_light"),
              marginTop: width * 0.012,
            }}
          >
            {tease}
          </div>
        ) : null}
        <div
          style={{
            fontFamily: bodyFont,
            fontSize: scalePx(24, width),
            letterSpacing: "0.26em",
            color: color("gold_light"),
            marginTop: width * 0.03,
            direction: "ltr",
          }}
        >
          NEXT STORY · SUBSCRIBE · FOLLOW
        </div>
      </div>
    </AbsoluteFill>
  );
};
