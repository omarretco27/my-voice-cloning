import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { useTheme, scalePx } from "../engine/theme";
import { textTiming } from "../engine/timing";

/** B — كارت الفصل: «الفصل الأول» + العنوان */
export const ChapterCard: React.FC<{ kicker?: string; title: string; durFrames: number }> = ({
  kicker = "الفصل",
  title,
  durFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { color, headlineFont, bodyFont, width } = useTheme();
  const t = textTiming(frame, fps, 10, durFrames - 8, width * 0.01);
  const lineW = width * 0.12;

  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
      <div style={{ opacity: t.opacity, transform: `translateY(${t.translateY}px)`, textAlign: "center" }}>
        <div
          style={{
            fontFamily: bodyFont,
            fontSize: scalePx(40, width),
            letterSpacing: "0.28em",
            color: color("stone_light"),
            marginBottom: width * 0.016,
          }}
        >
          {kicker}
        </div>
        <div
          style={{
            fontFamily: headlineFont,
            fontWeight: 700,
            fontSize: scalePx(78, width),
            color: color("parchment"),
          }}
        >
          {title}
        </div>
        <div
          style={{
            width: lineW,
            height: 1,
            background: `linear-gradient(90deg, transparent, ${color("gold")}, transparent)`,
            margin: `${width * 0.028}px auto 0`,
          }}
        />
      </div>
    </AbsoluteFill>
  );
};
