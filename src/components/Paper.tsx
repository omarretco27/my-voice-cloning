import React from "react";
import { AbsoluteFill } from "remotion";
import { useTheme } from "../engine/theme";

/**
 * طبقة الـLook الثابتة: نسيج ورق + حبيبات + تظليل + إطار ذهبي رفيع.
 * كل القيم من brand.json — لا أرقام في الكود.
 */
export const Paper: React.FC<{ children?: React.ReactNode; scale?: number }> = ({
  children,
  scale = 1,
}) => {
  const { color, brand } = useTheme();
  const look = brand.theme.look;

  return (
    <AbsoluteFill style={{ backgroundColor: color("ink_deep") }}>
      {/* خلفية دافئة + عمق */}
      <AbsoluteFill
        style={{
          transform: `scale(${scale})`,
          background: `
            radial-gradient(1200px 700px at 50% -6%, ${color("ink_second")} 0%, transparent 62%),
            radial-gradient(900px 800px at 50% 112%, #2a1f10 0%, transparent 58%),
            linear-gradient(180deg, #131109 0%, ${color("ink_deep")} 70%)`,
        }}
      />
      {/* نسيج الورق */}
      <AbsoluteFill
        style={{
          opacity: look.paper_texture?.opacity ?? 0.1,
          mixBlendMode: "soft-light" as any,
          backgroundImage: `repeating-linear-gradient(0deg, rgba(255,255,255,.05) 0 2px, transparent 2px 5px),
                            repeating-linear-gradient(90deg, rgba(255,255,255,.03) 0 3px, transparent 3px 7px)`,
        }}
      />
      {/* حبيبات */}
      <AbsoluteFill style={{ opacity: look.grain ?? 0.12 }}>
        <svg width="100%" height="100%">
          <filter id="riwaq-grain">
            <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch" />
            <feColorMatrix type="saturate" values="0" />
          </filter>
          <rect width="100%" height="100%" filter="url(#riwaq-grain)" opacity="0.5" />
        </svg>
      </AbsoluteFill>
      {/* تظليل */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(closest-side at 50% 45%, transparent 55%, rgba(0,0,0,${look.vignette ?? 0.18}) 100%)`,
        }}
      />
      {children}
    </AbsoluteFill>
  );
};

/** إطار ذهبي رفيع — موضع الإطار منفصل عن محتوى المكوّنات */
export const GoldFrame: React.FC<{ insetPx?: number }> = ({ insetPx }) => {
  const { color, width } = useTheme();
  const inset = insetPx ?? width * 0.028;
  return (
    <AbsoluteFill
      style={{
        margin: inset,
        border: `${Math.max(1, width * 0.0012)}px solid ${color("gold")}55`,
        borderRadius: width * 0.006,
      }}
    />
  );
};
