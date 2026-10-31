import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { useTheme, scalePx } from "../engine/theme";
import { textTiming } from "../engine/timing";

/** C — كارت اقتباس: Amiri Regular بمسند ذهبي عمودي */
export const QuoteCard: React.FC<{ text: string; source?: string; durFrames: number }> = ({
  text,
  source,
  durFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { color, headlineFont, bodyFont, width, safe } = useTheme();
  const t = textTiming(frame, fps, 12, durFrames - 10, width * 0.012);

  return (
    <AbsoluteFill
      style={{ alignItems: "center", justifyContent: "center", paddingLeft: safe.side, paddingRight: safe.side }}
    >
      <div
        style={{
          opacity: t.opacity,
          transform: `translateY(${t.translateY}px)`,
          borderRight: `${Math.max(2, width * 0.002)}px solid ${color("gold")}99`,
          paddingRight: width * 0.026,
          maxWidth: width * 0.8,
        }}
      >
        <div
          style={{
            fontFamily: headlineFont,
            fontWeight: 400,
            fontSize: scalePx(66, width),
            lineHeight: 1.7,
            color: "#F1E9D8",
          }}
        >
          {text}
        </div>
        {source ? (
          <div
            style={{
              fontFamily: bodyFont,
              fontSize: scalePx(30, width),
              color: color("gold_light"),
              marginTop: width * 0.018,
            }}
          >
            — {source}
          </div>
        ) : null}
      </div>
    </AbsoluteFill>
  );
};
