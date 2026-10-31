import React from "react";
import { AbsoluteFill } from "remotion";
import { useTheme } from "../engine/theme";
import { BrandMark } from "./BrandMark";

/** العلامة المائية — أعلى الوسط، ثابتة، شفافية من الثيم */
export const Watermark: React.FC = () => {
  const { brand, width, color } = useTheme();
  const wm = brand.theme.watermark;
  if (!wm) return null;
  const size = (wm.size_px * width) / 1080;
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "flex-start" }}>
      <div style={{ marginTop: width * 0.055, opacity: wm.opacity }}>
        <BrandMark sizePx={size} color={color("gold_light")} />
      </div>
    </AbsoluteFill>
  );
};
