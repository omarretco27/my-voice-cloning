import React, { createContext, useContext } from "react";
import type { Brand, SafeZone, AspectRatio } from "./types";

// ═══════════ مزود الثيم — القيم كلها من brand.json ولا شيء من الكود ═══════════

export interface ThemeCtx {
  brand: Brand;
  width: number;
  height: number;
  aspect: AspectRatio;
  safe: SafeZone;
  color: (key: string) => string;
  headlineFont: string;
  bodyFont: string;
}

const Ctx = createContext<ThemeCtx | null>(null);

const REF: Record<AspectRatio, { w: number; h: number; scaleRef: number }> = {
  "9x16": { w: 1080, h: 1920, scaleRef: 1080 },
  "16x9": { w: 1920, h: 1080, scaleRef: 1920 },
  "1x1": { w: 1080, h: 1080, scaleRef: 1080 },
  "4x5": { w: 1080, h: 1350, scaleRef: 1080 },
};

export function computeSafeZone(brand: Brand, aspect: AspectRatio, width: number, height: number): SafeZone {
  const keyMap: Record<AspectRatio, string> = {
    "9x16": "9x16_1080x1920",
    "16x9": "16x9_1920x1080",
    "1x1": "1x1_1080x1080",
    "4x5": "4x5_1080x1350",
  };
  const raw = brand.theme.safe_zones[keyMap[aspect]];
  const s = width / REF[aspect].scaleRef;
  return {
    top: raw.top_px * s,
    bottom: raw.bottom_px * s,
    side: raw.side_px * s,
    captionY: [raw.caption_band_y[0] * s, raw.caption_band_y[1] * s] as [number, number],
    width,
    height,
  };
}

export function aspectOf(width: number, height: number, meta: AspectRatio[]): AspectRatio {
  const ratio = width / height;
  if (Math.abs(ratio - 9 / 16) < 0.01) return "9x16";
  if (Math.abs(ratio - 16 / 9) < 0.01) return "16x9";
  if (Math.abs(ratio - 1) < 0.01) return "1x1";
  return "4x5";
}

export const ThemeProvider: React.FC<{
  brand: Brand;
  width: number;
  height: number;
  aspect: AspectRatio;
  children: React.ReactNode;
}> = ({ brand, width, height, aspect, children }) => {
  const safe = computeSafeZone(brand, aspect, width, height);
  const value: ThemeCtx = {
    brand,
    width,
    height,
    aspect,
    safe,
    color: (key: string) => brand.theme.colors[key]?.hex ?? "#FF00FF",
    headlineFont: `"${brand.theme.fonts.headline.family}", serif`,
    bodyFont: `"${brand.theme.fonts.body.family}", sans-serif`,
  };
  return React.createElement(Ctx.Provider, { value }, children);
};

export function useTheme(): ThemeCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error("useTheme خارج ThemeProvider");
  return v;
}

/** مقياس نص: القيم الأساسية مضبوطة على 1080 عرضًا ثم تتدرج مع المقاس الفعلي */
export function scalePx(px: number, width: number): number {
  return (px * width) / 1080;
}
