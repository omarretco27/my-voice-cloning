import React from "react";

/** المَرْكيزة الرسمية — قوس رواق + الجوهرة الذهبية (من أصول الحزمة) */
export const BrandMark: React.FC<{ sizePx: number; color?: string; jewel?: string }> = ({
  sizePx,
  color,
  jewel,
}) => (
  <svg
    viewBox="0 0 111.48 104.98"
    style={{ width: sizePx, height: (sizePx * 104.98) / 111.48, display: "block" }}
  >
    <g transform="translate(-4.26,-8.76)">
      <path
        d="M24 104L24 58C24.00 37.04 39.28 19.21 60.00 16.00C80.72 19.21 96.00 37.04 96.00 58.00L96 104ZM37 104L37 58C37.00 44.57 46.75 33.13 60.00 31.00C73.25 33.13 83.00 44.57 83.00 58.00L83 104Z"
        fill={color ?? "currentColor"}
        fillRule="evenodd"
      />
      <path d="M14 104H106" fill="none" stroke={color ?? "currentColor"} strokeWidth={5} strokeLinecap="round" />
      <circle cx="60" cy="66" r="6.5" fill={jewel ?? "#C2A469"} />
    </g>
  </svg>
);

/** الشعار الرأسي (رمز + «رِواق» + «تَارِيخْ وَتُرَاث») */
export const BrandLockup: React.FC<{ sizePx: number; color?: string; tagline?: string }> = ({
  sizePx,
  color,
  tagline = "تَارِيخْ وَتُرَاث",
}) => (
  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: sizePx * 0.06 }}>
    <BrandMark sizePx={sizePx * 0.62} color={color ?? "currentColor"} />
    <div
      style={{
        fontFamily: '"Amiri", serif',
        fontWeight: 700,
        fontSize: sizePx * 0.5,
        color: color ?? "currentColor",
        lineHeight: 1,
      }}
    >
      رِواق
    </div>
    <div style={{ fontFamily: '"Amiri", serif', fontSize: sizePx * 0.15, color: color ?? "currentColor", opacity: 0.75 }}>
      {tagline}
    </div>
  </div>
);
