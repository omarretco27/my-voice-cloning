import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { useTheme, scalePx } from "../engine/theme";
import { textTiming } from "../engine/timing";

/** A — إطار الهوك: عنوان افتتاحي (سطران أقصى) + كيكر + نقطة ذهبية */
export const HookFrame: React.FC<{ kicker?: string; title: string; durFrames: number }> = ({
  kicker,
  title,
  durFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { color, headlineFont, bodyFont, width, safe } = useTheme();
  const { opacity, translateY } = textTiming(frame, fps, 12, durFrames - 10, width * 0.014);

  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", padding: safe.side }}>
      <div style={{ transform: `translateY(${translateY}px)`, opacity, textAlign: "center" }}>
        {kicker ? (
          <div
            style={{
              fontFamily: bodyFont,
              fontWeight: 700,
              fontSize: scalePx(44, width),
              color: color("gold_light"),
              marginBottom: width * 0.022,
            }}
          >
            {kicker}
          </div>
        ) : null}
        <div
          style={{
            fontFamily: headlineFont,
            fontWeight: 700,
            fontSize: scalePx(96, width),
            lineHeight: 1.5,
            color: color("parchment"),
            whiteSpace: "pre-line",
          }}
        >
          {title}
        </div>
        <div
          style={{
            width: width * 0.008,
            height: width * 0.008,
            borderRadius: "50%",
            background: color("gold_light"),
            margin: `${width * 0.03}px auto 0`,
          }}
        />
      </div>
    </AbsoluteFill>
  );
};
