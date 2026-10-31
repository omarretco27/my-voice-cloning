import React from "react";
import { Composition } from "remotion";
import { Episode } from "./compositions/Episode";
import type { Brand, EpisodeMeta } from "./engine/types";
import brandJson from "../data/brand.json";
import goldenJson from "../data/projects/GOLDEN_001/episode_meta.json";

const brand = brandJson as unknown as Brand;
const golden = goldenJson as unknown as EpisodeMeta;

const lastSceneEnd = (meta: EpisodeMeta) =>
  Math.max(
    meta.edit?.duration_s ?? 0,
    ...(meta.story?.scenes ?? []).map((s) => s.t_out),
    1
  );

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Episode"
      component={Episode as React.ComponentType<any>}
      durationInFrames={30 * 42}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={{ meta: golden, brand }}
      calculateMetadata={({ props }: any) => {
        const meta: EpisodeMeta = props.meta;
        const fps = meta.edit?.fps ?? 30;
        const [w, h] = meta.render?.resolution ?? [1080, 1920];
        return {
          fps,
          width: w,
          height: h,
          durationInFrames: Math.round(lastSceneEnd(meta) * fps),
        };
      }}
    />
  );
};
