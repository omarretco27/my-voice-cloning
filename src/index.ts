// ملاحظة: استيراد الخطوط يجب أن يسبق registerRoot لضمان توفرها قبل أول إطار
import "./engine/fonts";
import { registerRoot } from "remotion";
import { RemotionRoot } from "./Root";

registerRoot(RemotionRoot);
