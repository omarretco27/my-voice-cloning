import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { useTheme, scalePx } from "../engine/theme";
import { textTiming } from "../engine/timing";

/** E — لور-ثِرد: اسم المتحدث ودوره، مسند ذهبي يمين، ثابت في مساره */
export const LowerThird: React.FC<{ name: string; role?: string; durFrames: number }> = ({
  name,
  role,
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
          bottom: safe.bottom * 0.42,
          right: safe.side,
          borderRight: `${Math.max(2, width * 0.0022)}px solid ${color("gold")}`,
          paddingRight: width * 0.018,
          textAlign: "right",
          opacity: t.opacity,
          transform: `translateY(${t.translateY}px)`,
        }}
      >
        <div style={{ fontFamily: bodyFont, fontWeight: 700, fontSize: scalePx(40, width), color: color("parchment") }}>
          {name}
        </div>
        {role ? (
          <div
            style={{
              fontFamily: bodyFont,
              fontWeight: 400,
              fontSize: scalePx(30, width),
              color: color("stone_light"),
              marginTop: width * 0.004,
            }}
          >
            {role}
          </div>
        ) : null}
      </div>
    </AbsoluteFill>
  );
};
