from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

try:
    from tqdm.auto import tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, n: int = 1) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable=None, **kwargs):
        return _TqdmFallback(iterable=iterable, **kwargs)


NAIP_IMAGE_ROOT = Path("/media/disk5/dataset/satlas/dataset/naip")
AUTO_MASK_LABEL_ROOT = Path("/media/disk5/dataset/satlas/sam3_auto_masks")
DEFAULT_OUTPUT_DIR = Path("/media/disk5/dataset/satlas/sam3_auto_mask_visualizations")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
HEADER_BACKGROUND = (22, 22, 22)
CANVAS_BACKGROUND = (248, 248, 248)
MASK_COLORS = (
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (231, 111, 81),
    (118, 200, 147),
    (106, 76, 147),
    (255, 107, 107),
    (78, 205, 196),
    (58, 134, 255),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SAM3 automatic-mask prediction results."
    )
    parser.add_argument("--label-root", type=Path, default=AUTO_MASK_LABEL_ROOT)
    parser.add_argument("--naip-root", type=Path, default=NAIP_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.label_root.exists():
        raise FileNotFoundError(f"Label root does not exist: {args.label_root}")
    if not args.naip_root.exists():
        raise FileNotFoundError(f"NAIP root does not exist: {args.naip_root}")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be a positive integer")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")


def scan_prediction_files(label_root: Path) -> list[Path]:
    json_paths: list[str] = []
    pending_dirs = [str(label_root)]

    while pending_dirs:
        current_dir = pending_dirs.pop()
        with os.scandir(current_dir) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending_dirs.append(entry.path)
                    continue
                if entry.is_file(follow_symlinks=False) and entry.name.endswith(".json"):
                    json_paths.append(entry.path)

    return [Path(path) for path in sorted(json_paths)]


def load_prediction(label_path: Path) -> dict[str, Any]:
    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Prediction file must contain a JSON object: {label_path}")
    return data


def unique_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def resolve_image_path(
    label_path: Path,
    label_root: Path,
    naip_root: Path,
    image_file: str | None,
) -> Path:
    candidates: list[Path] = []

    if image_file:
        image_rel_path = Path(image_file)
        candidates.append(naip_root / image_rel_path)
        if len(image_rel_path.parts) >= 2:
            candidates.append(
                naip_root / image_rel_path.parts[0] / "tci" / Path(*image_rel_path.parts[1:])
            )

    label_relative_parent = label_path.relative_to(label_root).parent
    for extension in IMAGE_EXTENSIONS:
        candidates.append(naip_root / label_relative_parent / f"{label_path.stem}{extension}")
        if "tci" in label_relative_parent.parts:
            without_tci = Path(*[part for part in label_relative_parent.parts if part != "tci"])
            candidates.append(naip_root / without_tci / f"{label_path.stem}{extension}")

    for candidate in unique_paths(candidates):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Unable to locate source image for prediction file: {label_path}"
    )


def decode_rle_mask(rle: dict[str, Any]) -> np.ndarray:
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def build_visible_masks(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    visible_items: list[dict[str, Any]] = []

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, dict):
            continue
        if "counts" not in segmentation or "size" not in segmentation:
            continue

        decoded_mask = decode_rle_mask(segmentation)
        area = int(decoded_mask.sum())
        if area <= 0:
            continue

        visible_items.append(
            {
                "mask": decoded_mask,
                "area": area,
                "predicted_iou": float(annotation.get("predicted_iou", 0.0)),
                "stability_score": float(annotation.get("stability_score", 0.0)),
            }
        )

    return sorted(
        visible_items,
        key=lambda item: (item["area"], item["predicted_iou"], item["stability_score"]),
        reverse=True,
    )


def color_for_mask_index(mask_index: int) -> tuple[int, int, int]:
    return MASK_COLORS[mask_index % len(MASK_COLORS)]


def apply_mask_overlay(
    image_rgb: np.ndarray,
    visible_items: list[dict[str, Any]],
    alpha: float,
) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()

    for mask_index, item in enumerate(visible_items):
        color = np.array(color_for_mask_index(mask_index), dtype=np.float32)
        mask = item["mask"]
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_visualization(
    image_path: Path,
    image_file: str,
    visible_items: list[dict[str, Any]],
    alpha: float,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image_rgb = np.asarray(image, dtype=np.uint8)
    overlay_rgb = apply_mask_overlay(image_rgb, visible_items, alpha=alpha)

    header_height = 56
    width, height = image.size
    canvas = Image.new(
        "RGB",
        (width * 2, height + header_height),
        CANVAS_BACKGROUND,
    )
    canvas.paste(image, (0, header_height))
    canvas.paste(Image.fromarray(overlay_rgb).convert("RGB"), (width, header_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width * 2, header_height), fill=HEADER_BACKGROUND)
    draw.text((12, 8), "Original", fill=(255, 255, 255), font=font)
    draw.text((width + 12, 8), "SAM3 Auto Masks", fill=(255, 255, 255), font=font)
    draw.text((12, 30), image_file, fill=(220, 220, 220), font=font)
    draw.text(
        (width + 12, 30),
        f"Kept masks: {len(visible_items)}",
        fill=(220, 220, 220),
        font=font,
    )
    return canvas


def relative_output_path(label_path: Path, label_root: Path) -> Path:
    return label_path.relative_to(label_root).with_suffix(".png")


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)

    label_files = scan_prediction_files(args.label_root)
    if not label_files:
        raise FileNotFoundError(f"No prediction json files found under: {args.label_root}")

    random_generator = random.Random(args.seed)
    random_generator.shuffle(label_files)

    target_count = min(args.num_samples, len(label_files))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved_items: list[dict[str, Any]] = []
    failed_items: list[dict[str, str]] = []
    progress_bar = tqdm(total=target_count, desc="Visualizing SAM3 auto masks", unit="image")

    for label_path in label_files:
        if len(saved_items) >= target_count:
            break

        try:
            prediction = load_prediction(label_path)
            image_file = prediction.get("image_file")
            image_path = resolve_image_path(
                label_path=label_path,
                label_root=args.label_root,
                naip_root=args.naip_root,
                image_file=image_file if isinstance(image_file, str) else None,
            )
            if not isinstance(image_file, str) or not image_file:
                image_file = image_path.relative_to(args.naip_root).as_posix()

            annotations = prediction.get("annotation", [])
            if not isinstance(annotations, list):
                raise ValueError(f"`annotation` must be a list: {label_path}")

            visible_items = build_visible_masks(annotations)
            if not visible_items:
                continue

            visualization = render_visualization(
                image_path=image_path,
                image_file=image_file,
                visible_items=visible_items,
                alpha=args.alpha,
            )
            output_path = args.output_dir / relative_output_path(label_path, args.label_root)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            visualization.save(output_path)

            saved_items.append(
                {
                    "label_file": str(label_path),
                    "image_file": image_file,
                    "image_path": str(image_path),
                    "visualization_file": str(output_path),
                    "num_masks": len(visible_items),
                }
            )
            progress_bar.update(1)
        except Exception as exc:
            failed_items.append(
                {
                    "label_file": str(label_path),
                    "error": str(exc),
                }
            )

    progress_bar.close()
    dump_json(saved_items, args.output_dir / "visualization_manifest.json")
    if failed_items:
        dump_json(failed_items, args.output_dir / "visualization_failures.json")

    print(f"Saved {len(saved_items)} visualization images to: {args.output_dir}")
    print(f"Visualization manifest: {args.output_dir / 'visualization_manifest.json'}")
    if failed_items:
        print(f"Failed files log: {args.output_dir / 'visualization_failures.json'}")


if __name__ == "__main__":
    main()
