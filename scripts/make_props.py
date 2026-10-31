#!/usr/bin/env python3
"""make_props.py — يولّد ملف props موحّد للريندر: { meta, brand }
السبب: واجهة التركيب تستقبل props = { meta, brand }، وملف episode_meta وحده لا يكفي.
الاستخدام: python scripts/make_props.py data/projects/GOLDEN_001
"""
import json, os, sys


def build(project_dir: str) -> str:
    meta_path = os.path.join(project_dir, "episode_meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"✘ غير موجود: {meta_path}")
    meta = json.load(open(meta_path, encoding="utf-8"))
    brand_path = (meta.get("brand_ref") or {}).get("theme_path", "data/brand.json")
    if not os.path.exists(brand_path):
        raise SystemExit(f"✘ ملف الثيم غير موجود: {brand_path}")
    brand = json.load(open(brand_path, encoding="utf-8"))
    out = os.path.join(project_dir, "render_props.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "brand": brand}, f, ensure_ascii=False, indent=2)
    print(f"✔ {out}  (project={meta['project']['id']} · brand={brand.get('brand_id')})")
    return out


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/projects/GOLDEN_001"
    build(target)
