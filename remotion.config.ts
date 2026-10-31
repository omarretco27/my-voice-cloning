import { Config } from "@remotion/cli/config";

// ── إعدادات متوافقة مع جهاز 8GB RAM / 2GB VRAM ──
Config.setVideoImageFormat("jpeg");
Config.setJpegQuality(90);
Config.setConcurrency(1);
Config.overrideWebpackConfig((cfg) => cfg);
Config.setChromiumOpenGlRenderer("swiftshader");
Config.setCodec("h264");
Config.setCrf(18);
Config.setAudioCodec("aac");
Config.setPixelFormat("yuv420p");
Config.setOverwriteOutput(true);
