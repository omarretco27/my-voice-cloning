// ── أنواع النظام: تطابق schemas/episode_meta.schema.json ──

export type AspectRatio = "9x16" | "16x9" | "1x1" | "4x5";

export interface BrandColor {
  hex: string;
  role: string;
  max_ratio?: number;
}

export interface Brand {
  schema_version: string;
  brand_id: string;
  name_ar: string;
  tagline_ar: string;
  theme: {
    colors: Record<string, BrandColor>;
    fonts: {
      headline: { family: string };
      body: { family: string };
      forbidden?: string[];
    };
    type_scale_1080x1920: Record<string, any>;
    logo: Record<string, any>;
    watermark: { asset: string; opacity: number; position: string; size_px: number };
    look: { paper_texture: any; grain: number; vignette: number; bloom: number };
    motion: {
      ease: string;
      text_in_ms: number;
      text_out_ms: number;
      rise_px: number;
      ken_burns: { from: number; to: number };
      cut_rhythm_s: Record<string, number[]>;
      transitions: string[];
    };
    safe_zones: Record<string, SafeZoneRaw>;
    captions?: any;
    negative_rules: string[];
  };
  sound: any;
  video_type_presets?: Record<string, any>;
  platform_seo_rules?: any;
}

export interface SafeZoneRaw {
  top_px: number;
  bottom_px: number;
  side_px: number;
  caption_band_y: [number, number];
}

export interface SafeZone {
  top: number;
  bottom: number;
  side: number;
  captionY: [number, number];
  width: number;
  height: number;
}

export type SceneType =
  | "hook"
  | "chapter"
  | "quote"
  | "stat"
  | "lower_third"
  | "source"
  | "end"
  | "broll";

export interface Scene {
  type: SceneType;
  t_in: number;
  t_out: number;
  // hook / chapter
  title?: string;
  kicker?: string;
  // quote
  text?: string;
  source?: string;
  // stat
  number?: string;
  label?: string;
  // lower_third
  name?: string;
  role?: string;
  // source
  value?: string;
}

export interface CaptionLine {
  t_in: number;
  t_out: number;
  ar: string;
  en?: string;
  keywords?: string[];
}

export interface EpisodeMeta {
  schema_version: string;
  brand_ref: { theme_path: string; watermark: "on" | "off" };
  project: {
    id: string;
    series_id?: string | null;
    title: string;
    episode_number?: number | null;
    video_type: "short" | "explainer" | "documentary" | "podcast_clip" | "promo" | "product" | "story" | "news";
    platforms: string[];
    aspect_ratios: AspectRatio[];
    language_primary: string;
    language_secondary?: string | null;
    priority?: "speed" | "quality" | "balanced";
  };
  source?: any;
  story?: {
    hook?: string;
    chapters?: { title: string; t_in: number; t_out: number }[];
    scenes?: Scene[];
    cta_question?: string;
    next_tease?: string;
  };
  narration?: {
    mode: "generate" | "provided" | "none";
    profile_id?: string;
    placement_map?: { t_in: number; t_out: number; type: string }[];
  };
  audio?: {
    vo?: { src: string; t_in: number; t_out?: number; volume_db?: number }[];
    music?: { src: string; t_in: number; t_out?: number; volume_db?: number } | null;
    stingers?: { src: string; t_in: number; volume_db?: number }[];
  };
  captions?: {
    styles: string[];
    languages: string[];
    sync_source: string;
    data: CaptionLine[];
  } | null;
  edit?: {
    fps: number;
    cut_style: string;
    duration_s?: number;
    music_mood?: string;
    stingers?: { in: boolean; out: boolean };
  };
  render?: { resolution: [number, number]; preset: string; per_platform_export: boolean };
  deliverables: Record<string, any>;
  qc_rules?: { must_pass: string[]; tolerance_lu: number };
  status?: string;
  notes?: string | null;
}
