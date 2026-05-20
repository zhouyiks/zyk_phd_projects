from __future__ import annotations

import argparse
import json
import math
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
SAM3_LABEL_ROOT = Path("/media/disk5/dataset/satlas/sam3_labels")
DEFAULT_OUTPUT_DIR = Path("/media/disk5/dataset/satlas/sam3_visualizations")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
PANEL_BACKGROUND_COLOR = (248, 248, 248)
HEADER_BACKGROUND_COLOR = (22, 22, 22)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly visualize SAM3 NAIP prediction results."
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=SAM3_LABEL_ROOT,
        help="Root directory that stores SAM3 prediction json files.",
    )
    parser.add_argument(
        "--naip-root",
        type=Path,
        default=NAIP_IMAGE_ROOT,
        help="Root directory that stores NAIP images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save visualization images.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="How many prediction files are randomly visualized.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling prediction files.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Mask overlay alpha in [0, 1].",
    )
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

        if len(image_rel_path.parts) == 1:
            label_relative_parent = label_path.relative_to(label_root).parent
            candidates.append(naip_root / label_relative_parent / image_rel_path.name)

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


def build_class_panels(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    class_to_data: dict[str, dict[str, Any]] = {}

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue

        class_name = annotation.get("class_name")
        segmentation_list = annotation.get("segmentation")
        if not isinstance(class_name, str) or not isinstance(segmentation_list, list):
            continue

        for rle in segmentation_list:
            if not isinstance(rle, dict):
                continue
            if "counts" not in rle or "size" not in rle:
                continue

            decoded_mask = decode_rle_mask(rle)
            class_data = class_to_data.setdefault(
                class_name,
                {
                    "class_name": class_name,
                    "masks": [],
                },
            )
            if decoded_mask.any():
                class_data["masks"].append(decoded_mask)

    visible_items = []
    for class_data in class_to_data.values():
        if not class_data["masks"]:
            continue

        visible_mask = np.zeros_like(class_data["masks"][0], dtype=bool)
        for mask in class_data["masks"]:
            visible_mask |= mask

        area = int(visible_mask.sum())
        if area <= 0:
            continue

        visible_items.append(
            {
                "class_name": class_data["class_name"],
                "masks": sorted(
                    class_data["masks"],
                    key=lambda mask: int(mask.sum()),
                    reverse=True,
                ),
                "num_masks": len(class_data["masks"]),
                "area": area,
            }
        )

    return sorted(visible_items, key=lambda item: item["area"], reverse=True)


def color_for_mask_index(mask_index: int) -> tuple[int, int, int]:
    palette = (
        (230, 57, 70),
        (29, 53, 87),
        (69, 123, 157),
        (42, 157, 143),
        (38, 70, 83),
        (233, 196, 106),
        (244, 162, 97),
        (231, 111, 81),
        (118, 200, 147),
        (106, 76, 147),
        (255, 107, 107),
        (78, 205, 196),
    )
    return palette[mask_index % len(palette)]


def apply_mask_overlay(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    alpha: float,
) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()

    for mask_index, mask in enumerate(masks):
        color = np.array(color_for_mask_index(mask_index), dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_panel(
    image_rgb: np.ndarray,
    title: str,
    subtitle: str,
    masks: list[np.ndarray] | None,
    alpha: float,
) -> Image.Image:
    if masks:
        panel_rgb = apply_mask_overlay(image_rgb, masks, alpha=alpha)
    else:
        panel_rgb = image_rgb

    image = Image.fromarray(panel_rgb).convert("RGBA")
    width, height = image.size
    header_height = 54
    canvas = Image.new(
        "RGBA",
        (width, height + header_height),
        (*PANEL_BACKGROUND_COLOR, 255),
    )
    canvas.paste(image, (0, header_height))

    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, width, header_height), fill=(*HEADER_BACKGROUND_COLOR, 255))
    draw.text((12, 8), title, fill=(255, 255, 255, 255), font=font)
    draw.text((12, 28), subtitle, fill=(220, 220, 220, 255), font=font)
    return canvas.convert("RGB")


def render_visualization(
    image_path: Path,
    image_file: str,
    visible_items: list[dict[str, Any]],
    alpha: float,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image_rgb = np.asarray(image, dtype=np.uint8)
    width, height = image.size

    panels = [
        render_panel(
            image_rgb=image_rgb,
            title="Original",
            subtitle=image_file,
            masks=None,
            alpha=alpha,
        )
    ]
    for item in visible_items:
        panels.append(
            render_panel(
                image_rgb=image_rgb,
                title=item["class_name"],
                subtitle=f"{item['num_masks']} masks",
                masks=item["masks"],
                alpha=alpha,
            )
        )

    num_panels = len(panels)
    num_cols = max(1, min(4, math.ceil(math.sqrt(num_panels))))
    num_rows = math.ceil(num_panels / num_cols)
    panel_width, panel_height = panels[0].size
    gap = 16
    title_height = 60

    canvas = Image.new(
        "RGB",
        (
            num_cols * panel_width + (num_cols + 1) * gap,
            title_height + num_rows * panel_height + (num_rows + 1) * gap,
        ),
        PANEL_BACKGROUND_COLOR,
    )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((gap, 12), "SAM3 Per-Class Visualization", fill=(20, 20, 20), font=font)
    draw.text(
        (gap, 32),
        f"{image_file} | visible classes: {len(visible_items)}",
        fill=(70, 70, 70),
        font=font,
    )

    for panel_index, panel in enumerate(panels):
        row_idx = panel_index // num_cols
        col_idx = panel_index % num_cols
        offset_x = gap + col_idx * (panel_width + gap)
        offset_y = title_height + gap + row_idx * (panel_height + gap)
        canvas.paste(panel, (offset_x, offset_y))

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
    progress_bar = tqdm(total=target_count, desc="Visualizing SAM3", unit="image")

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

            visible_items = build_class_panels(annotations)
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
                    "num_visible_classes": len(visible_items),
                    "classes": [
                        {
                            "class_name": item["class_name"],
                            "num_masks": item["num_masks"],
                            "area": item["area"],
                        }
                        for item in visible_items
                    ],
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
