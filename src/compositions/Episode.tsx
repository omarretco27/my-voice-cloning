import React from "react";
import { AbsoluteFill, Audio, Sequence, staticFile, useVideoConfig } from "remotion";
import type { Brand, EpisodeMeta, Scene } from "../engine/types";
import { ThemeProvider, aspectOf, useTheme } from "../engine/theme";
import { secToFrames } from "../engine/timing";
import { Paper, GoldFrame } from "../components/Paper";
import { Watermark } from "../components/Watermark";
import { Captions } from "../components/Captions";
import { HookFrame } from "../components/HookFrame";
import { ChapterCard } from "../components/ChapterCard";
import { QuoteCard } from "../components/QuoteCard";
import { StatCard } from "../components/StatCard";
import { LowerThird } from "../components/LowerThird";
import { SourceCard } from "../components/SourceCard";
import { EndScreen } from "../components/EndScreen";

const dbToVol = (db = 0) => Math.pow(10, db / 20);

const SceneRenderer: React.FC<{ scene: Scene; durFrames: number }> = ({ scene, durFrames }) => {
  switch (scene.type) {
    case "hook":
      return <HookFrame kicker={scene.kicker} title={scene.title ?? ""} durFrames={durFrames} />;
    case "chapter":
      return <ChapterCard kicker={scene.kicker} title={scene.title ?? ""} durFrames={durFrames} />;
    case "quote":
      return <QuoteCard text={scene.text ?? ""} source={scene.source} durFrames={durFrames} />;
    case "stat":
      return <StatCard number={scene.number ?? ""} label={scene.label ?? ""} durFrames={durFrames} />;
    case "lower_third":
      return <LowerThird name={scene.name ?? ""} role={scene.role} durFrames={durFrames} />;
    case "source":
      return <SourceCard label={scene.value ? scene.label : undefined} value={scene.value ?? ""} durFrames={durFrames} />;
    case "end":
      return <EndScreen durFrames={durFrames} />;
    default:
      return null;
  }
};

const Timeline: React.FC<{ meta: EpisodeMeta }> = ({ meta }) => {
  const { fps } = useVideoConfig();
  const { safe, brand } = useTheme();
  const scenes = meta.story?.scenes ?? [];
  const total = Math.round((meta.edit?.duration_s ?? 30) * fps);

  return (
    <AbsoluteFill>
      {/* طبقة الـLook: ثابتة خلف كل المشاهد */}
      <Paper />

      {/* المشاهد A–G */}
      {scenes.map((s, i) => {
        const from = secToFrames(s.t_in, fps);
        const dur = secToFrames(s.t_out - s.t_in, fps);
        return (
          <Sequence key={i} from={from} durationInFrames={dur} name={`${s.type}@${s.t_in}s`}>
            <SceneRenderer scene={s} durFrames={dur} />
          </Sequence>
        );
      })}

      {/* الكابشنز — مستقلة عن المشاهد، من البيانات */}
      {(meta.captions?.data ?? []).map((line, i) => {
        const from = secToFrames(line.t_in, fps);
        const dur = secToFrames(line.t_out - line.t_in, fps);
        return (
          <Sequence key={`c${i}`} from={from} durationInFrames={dur} name={`caption ${i + 1}`}>
            <Captions line={line} style={(meta.captions?.styles?.[0] as any) ?? "keyword_highlight"} />
          </Sequence>
        );
      })}

      {/* الإطار الذهبي + العلامة المائية */}
      <GoldFrame />
      {meta.brand_ref.watermark === "on" ? <Watermark /> : null}

      {/* الصوت: التعليق الرسمي + الموسيقى + النغمات (من الإعدادات) */}
      {(meta.audio?.vo ?? []).map((v, i) => (
        <Sequence key={`vo${i}`} from={secToFrames(v.t_in, fps)} name={`VO ${i + 1}`}>
          <Audio src={staticFile(v.src)} volume={dbToVol(v.volume_db)} />
        </Sequence>
      ))}
      {meta.audio?.music ? (
        <Audio
          src={staticFile(meta.audio.music.src)}
          volume={dbToVol(meta.audio.music.volume_db)}
          startFrom={0}
        />
      ) : null}
      {(meta.audio?.stingers ?? []).map((s, i) => (
        <Sequence key={`st${i}`} from={secToFrames(s.t_in, fps)} name={`stinger ${i + 1}`}>
          <Audio src={staticFile(s.src)} volume={dbToVol(s.volume_db)} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

export const Episode: React.FC<{ meta: EpisodeMeta; brand: Brand }> = ({ meta, brand }) => {
  const { width, height } = useVideoConfig();
  const aspect = aspectOf(width, height, meta.project.aspect_ratios);
  return (
    <ThemeProvider brand={brand} width={width} height={height} aspect={aspect}>
      <Timeline meta={meta} />
    </ThemeProvider>
  );
};
