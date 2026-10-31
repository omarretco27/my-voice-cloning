import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig, spring } from "remotion";
import { useTheme, scalePx } from "../engine/theme";
import { textTiming } from "../engine/timing";

/** D — كارت رقم: الرقم Amiri Bold ذهبيًا + سطر شرح عاجي (بلا عدّاد صارخ) */
export const StatCard: React.FC<{ number: string; label: string; durFrames: number }> = ({
  number,
  label,
  durFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { color, headlineFont, bodyFont, width } = useTheme();
  const t = textTiming(frame, fps, 12, durFrames - 10, width * 0.012);
  const grow = spring({ frame, fps, config: { damping: 22, stiffness: 90, mass: 0.9 } });

  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
      <div style={{ opacity: t.opacity, transform: `translateY(${t.translateY}px)`, textAlign: "center" }}>
        <div
          style={{
            fontFamily: headlineFont,
            fontWeight: 700,
            fontSize: scalePx(170, width) * (0.94 + 0.06 * grow),
            color: color("gold_light"),
            lineHeight: 1,
          }}
        >
          {number}
        </div>
        <div
          style={{
            fontFamily: bodyFont,
            fontWeight: 700,
            fontSize: scalePx(42, width),
            color: color("parchment"),
            marginTop: width * 0.02,
          }}
        >
          {label}
        </div>
      </div>
    </AbsoluteFill>
  );
};
