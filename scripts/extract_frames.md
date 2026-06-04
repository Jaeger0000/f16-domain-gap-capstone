# Frame Extraction

Videos of physical scale-model F16 targets were converted to image frames using the
`video_to_png_convertor` tool: https://github.com/Jaeger0000/video_to_png_convertor

Guidelines followed:
- Sampled diverse, non-duplicate frames (avoid near-identical consecutive frames).
- Kept original resolution (1920x1080) — no downscaling — so the 1280 experiment
  carries real detail.
- Test videos (e.g. Ambar_Arka, Atolye) were set aside BEFORE extraction so that no
  test frame leaks into train/val (video-level split).

After extraction, frames were annotated (helper-model-assisted + manual verification)
and exported in YOLO format with a single class `F16`.
