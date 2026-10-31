import { interpolate, Easing } from "remotion";

export const secToFrames = (s: number, fps: number) => Math.round(s * fps);

/** دخول/خروج موحّد للنصوص (من قواعد الحركة في brand.json) */
export function textTiming(
  frame: number,
  fps: number,
  inFrames: number,
  outFrames: number,
  rise: number
) {
  const enterT = interpolate(frame, [0, inFrames], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.2, 0.8, 0.2, 1),
  });
  const exitT = interpolate(frame, [outFrames, 0], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.2, 0.8, 0.2, 1),
  });
  return {
    opacity: Math.min(enterT, exitT),
    translateY: (1 - enterT) * rise - (1 - exitT) * rise * 0.6,
  };
}

/** Ken Burns خفيف جدًا للخلفية */
export function kenBurns(frame: number, durationFrames: number, from = 1.0, to = 1.06) {
  const t = interpolate(frame, [0, durationFrames], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return from + (to - from) * t;
}
