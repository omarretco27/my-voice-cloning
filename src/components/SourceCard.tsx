import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { useTheme, scalePx } from "../engine/theme";
import { textTiming } from "../engine/timing";

/** F — كارت مصدر: قاعدة ذهبية رفيعة + تسمية + القيمة (أداة المصداقية) */
export const SourceCard: React.FC<{ label?: string; value: string; durFrames: number }> = ({
  label = "المصدر",
  value,
  durFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { color, bodyFont, width, safe } = useTheme();
  const t = textTiming(frame, fps, 10, durFrames - 8, width * 0.008);

  return (
    <AbsoluteFill style={{ justifyContent: "flex-start" }}>
      <div
        style={{
          position: "absolute",
          left: safe.side,
          right: safe.side,
          bottom: safe.bottom * 0.62,
          borderTop: `1px solid ${color("gold")}88`,
          paddingTop: width * 0.012,
          textAlign: "right",
          opacity: t.opacity,
          transform: `translateY(${t.translateY}px)`,
        }}
      >
        <div style={{ fontFamily: bodyFont, fontWeight: 700, fontSize: scalePx(30, width), color: color("gold_light") }}>
          {label}
        </div>
        <div
          style={{
            fontFamily: bodyFont,
            fontWeight: 500,
            fontSize: scalePx(29, width),
            color: color("stone_light"),
            lineHeight: 1.6,
          }}
        >
          {value}
        </div>
      </div>
    </AbsoluteFill>
  );
};
