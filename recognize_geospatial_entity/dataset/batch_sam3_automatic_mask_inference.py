from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = PROJECT_ROOT / "sam3_main"

if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.train.masks_ops import rle_encode


"""
目标：实现类似 SAM1 automatic mask generator 的 SAM3 无类别自动分割。

实现方式：
1. 使用 SAM3 的 `enable_inst_interactivity=True` 图像交互预测器。
2. 对整张图生成规则点网格，将每个前景点作为一个 prompt。
3. 对所有候选 mask 做 predicted IoU / stability / 面积过滤。
4. 再做 mask IoU NMS，返回整张图的 mask 列表，不带类别名称。

注意：
1. SAM3 当前没有直接暴露 SAM1 那种 automatic mask generator API。
2. 这个脚本采用“全图点网格采样”的方式近似实现自动模式。
"""


DEFAULT_POINTS_PER_SIDE = 32
DEFAULT_POINTS_BATCH_SIZE = 64
DEFAULT_PRED_IOU_THRESH = 0.88
DEFAULT_STABILITY_SCORE_THRESH = 0.95
DEFAULT_STABILITY_SCORE_OFFSET = 1.0
DEFAULT_MASK_THRESHOLD = 0.0
DEFAULT_MIN_MASK_AREA = 64
DEFAULT_MASK_NMS_THRESH = 0.7
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sam3_automatic_mask_outputs"
NAIP_IMAGE_ROOT = Path("/media/disk5/dataset/satlas/dataset/naip")
NAIP_IMAGE_INDEX_FILE = Path("/media/disk5/dataset/satlas/naip_images.json")
SUPPORTED_CHECKPOINT_SUFFIXES = {".pt", ".pth", ".bin"}
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def iter_image_files(input_path: Path) -> Iterator[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {input_path}")
        yield input_path
        return

    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def scan_image_files_fast(input_path: Path) -> list[Path]:
    image_paths: list[str] = []
    pending_dirs = [str(input_path)]

    while pending_dirs:
        current_dir = pending_dirs.pop()
        with os.scandir(current_dir) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending_dirs.append(entry.path)
                    continue

                if (
                    entry.is_file(follow_symlinks=False)
                    and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS
                ):
                    image_paths.append(entry.path)

    return [Path(path) for path in sorted(image_paths)]


def load_or_create_naip_image_paths() -> list[Path]:
    if NAIP_IMAGE_INDEX_FILE.exists():
        with NAIP_IMAGE_INDEX_FILE.open("r", encoding="utf-8") as f:
            cached_paths = json.load(f)

        if not isinstance(cached_paths, list) or not all(
            isinstance(path, str) for path in cached_paths
        ):
            raise ValueError(
                f"NAIP image index must be a JSON list of strings: {NAIP_IMAGE_INDEX_FILE}"
            )

        return [Path(path) for path in cached_paths]

    if not NAIP_IMAGE_ROOT.exists():
        raise FileNotFoundError(f"NAIP image root does not exist: {NAIP_IMAGE_ROOT}")

    image_paths = scan_image_files_fast(NAIP_IMAGE_ROOT)
    NAIP_IMAGE_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NAIP_IMAGE_INDEX_FILE.open("w", encoding="utf-8") as f:
        json.dump([str(path) for path in image_paths], f, ensure_ascii=False, indent=2)
    return image_paths


def resolve_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir() and input_path.resolve() == NAIP_IMAGE_ROOT.resolve():
        return load_or_create_naip_image_paths()
    return list(iter_image_files(input_path))


def resolve_output_json_path(
    image_path: Path,
    input_path: Path,
    output_dir: Path,
) -> Path:
    if input_path.is_dir():
        relative_parent = image_path.relative_to(input_path).parent
        return output_dir / relative_parent / f"{image_path.stem}.json"
    return output_dir / f"{image_path.stem}.json"


def resolve_image_file_field(image_path: Path, input_path: Path) -> str:
    try:
        naip_relative_path = image_path.relative_to(NAIP_IMAGE_ROOT)
    except ValueError:
        naip_relative_path = None

    if naip_relative_path is not None:
        relative_parts = [part for part in naip_relative_path.parts if part != "tci"]
        return Path(*relative_parts).as_posix()

    if input_path.is_dir():
        return image_path.relative_to(input_path).as_posix()

    return image_path.name


def resolve_bpe_path() -> str:
    return str(
        Path(sam3.__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    )


def resolve_hf_checkpoint_path(hf_ckpt_path: Path) -> Path:
    if not hf_ckpt_path.exists():
        raise FileNotFoundError(
            f"HuggingFace checkpoint path does not exist: {hf_ckpt_path}"
        )

    if hf_ckpt_path.is_file():
        if hf_ckpt_path.suffix.lower() not in SUPPORTED_CHECKPOINT_SUFFIXES:
            raise ValueError(
                "SAM3 only supports torch checkpoint files with suffix "
                f"{sorted(SUPPORTED_CHECKPOINT_SUFFIXES)}: {hf_ckpt_path}"
            )
        return hf_ckpt_path

    preferred_filenames = ("sam3.pt", "checkpoint.pt", "pytorch_model.bin")
    for filename in preferred_filenames:
        candidate = hf_ckpt_path / filename
        if candidate.is_file():
            return candidate

    for pattern in ("*.pt", "*.pth", "*.bin"):
        matches = sorted(hf_ckpt_path.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "No torch checkpoint file was found under HuggingFace checkpoint directory: "
        f"{hf_ckpt_path}"
    )


def build_inference_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def build_model(
    device: torch.device,
    hf_ckpt_path: Path,
    compile_model: bool = False,
):
    return build_sam3_image_model(
        bpe_path=resolve_bpe_path(),
        device=str(device),
        checkpoint_path=str(resolve_hf_checkpoint_path(hf_ckpt_path)),
        load_from_HF=False,
        compile=compile_model,
        enable_inst_interactivity=True,
    )


def generate_point_grid(
    image_height: int,
    image_width: int,
    points_per_side: int,
) -> np.ndarray:
    offset = 1.0 / (2 * points_per_side)
    x_coords = np.linspace(offset, 1.0 - offset, points_per_side, dtype=np.float32)
    y_coords = np.linspace(offset, 1.0 - offset, points_per_side, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    points[:, 0] *= image_width
    points[:, 1] *= image_height
    return points


def ensure_mask_batch_dims(
    masks: np.ndarray,
    scores: np.ndarray,
    logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if masks.ndim == 3:
        masks = masks[None, ...]
    if scores.ndim == 1:
        scores = scores[None, ...]
    if logits.ndim == 3:
        logits = logits[None, ...]
    return masks, scores, logits


def compute_stability_scores(
    mask_logits: np.ndarray,
    stability_score_offset: float,
) -> np.ndarray:
    flat_logits = mask_logits.reshape(mask_logits.shape[0], -1)
    area_intersection = (flat_logits > stability_score_offset).sum(axis=1).astype(
        np.float32
    )
    area_union = (flat_logits > -stability_score_offset).sum(axis=1).astype(np.float32)
    return np.where(area_union > 0, area_intersection / area_union, 1.0)


def mask_to_bbox_xywh(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

    x_min = float(xs.min())
    y_min = float(ys.min())
    x_max = float(xs.max())
    y_max = float(ys.max())
    return [x_min, y_min, float(x_max - x_min + 1), float(y_max - y_min + 1)]


def bbox_xywh_to_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def boxes_intersect(
    bbox1: Sequence[float],
    bbox2: Sequence[float],
) -> bool:
    x1_min, y1_min, x1_max, y1_max = bbox_xywh_to_xyxy(bbox1)
    x2_min, y2_min, x2_max, y2_max = bbox_xywh_to_xyxy(bbox2)
    return not (
        x1_max <= x2_min
        or x2_max <= x1_min
        or y1_max <= y2_min
        or y2_max <= y1_min
    )


def compute_mask_iou(
    mask1: np.ndarray,
    mask2: np.ndarray,
    area1: int,
    area2: int,
) -> float:
    intersection = int(np.logical_and(mask1, mask2).sum())
    if intersection == 0:
        return 0.0
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def apply_mask_nms(
    candidates: list[dict[str, Any]],
    mask_nms_thresh: float,
) -> list[dict[str, Any]]:
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (item["predicted_iou"], item["stability_score"], item["area"]),
        reverse=True,
    )
    kept_candidates: list[dict[str, Any]] = []

    for candidate in sorted_candidates:
        keep = True
        for kept_candidate in kept_candidates:
            if not boxes_intersect(candidate["bbox"], kept_candidate["bbox"]):
                continue

            mask_iou = compute_mask_iou(
                candidate["mask"],
                kept_candidate["mask"],
                candidate["area"],
                kept_candidate["area"],
            )
            if mask_iou > mask_nms_thresh:
                keep = False
                break

        if keep:
            kept_candidates.append(candidate)

    return kept_candidates


def serialize_mask_annotation(candidate: dict[str, Any]) -> dict[str, Any]:
    mask_tensor = torch.from_numpy(candidate["mask"][None, ...]).to(dtype=torch.bool)
    segmentation = rle_encode(mask_tensor, return_areas=True)[0]
    return {
        "segmentation": segmentation,
        "area": candidate["area"],
        "bbox": candidate["bbox"],
        "predicted_iou": candidate["predicted_iou"],
        "stability_score": candidate["stability_score"],
        "point_coords": candidate["point_coords"],
    }


def generate_automatic_masks_for_single_image(
    model,
    processor: Sam3Processor,
    image: Image.Image,
    points_per_side: int,
    points_batch_size: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    stability_score_offset: float,
    min_mask_area: int,
    mask_nms_thresh: float,
) -> list[dict[str, Any]]:
    image_width, image_height = image.size
    point_grid = generate_point_grid(
        image_height=image_height,
        image_width=image_width,
        points_per_side=points_per_side,
    )

    candidates: list[dict[str, Any]] = []
    inference_state = processor.set_image_batch([image])
    for start in range(0, len(point_grid), points_batch_size):
        point_batch = point_grid[start : start + points_batch_size]
        point_coords = point_batch[:, None, :].astype(np.float32)
        point_labels = np.ones((len(point_batch), 1), dtype=np.int32)

        masks_batch, predicted_ious_batch, mask_logits_batch = model.predict_inst_batch(
            inference_state,
            point_coords_batch=[point_coords],
            point_labels_batch=[point_labels],
            box_batch=None,
            multimask_output=True,
            return_logits=True,
        )
        masks, predicted_ious, mask_logits = ensure_mask_batch_dims(
            masks=masks_batch[0] > DEFAULT_MASK_THRESHOLD,
            scores=predicted_ious_batch[0],
            logits=mask_logits_batch[0],
        )

        for prompt_idx in range(masks.shape[0]):
            stability_scores = compute_stability_scores(
                mask_logits[prompt_idx],
                stability_score_offset=stability_score_offset,
            )
            for mask_idx in range(masks.shape[1]):
                mask = masks[prompt_idx, mask_idx].astype(bool)
                area = int(mask.sum())
                if area < min_mask_area:
                    continue

                predicted_iou = float(predicted_ious[prompt_idx, mask_idx])
                if predicted_iou < pred_iou_thresh:
                    continue

                stability_score = float(stability_scores[mask_idx])
                if stability_score < stability_score_thresh:
                    continue

                candidates.append(
                    {
                        "mask": mask,
                        "area": area,
                        "bbox": mask_to_bbox_xywh(mask),
                        "predicted_iou": predicted_iou,
                        "stability_score": stability_score,
                        "point_coords": point_batch[prompt_idx].astype(float).tolist(),
                    }
                )

    deduplicated_candidates = apply_mask_nms(
        candidates=candidates,
        mask_nms_thresh=mask_nms_thresh,
    )
    return [serialize_mask_annotation(candidate) for candidate in deduplicated_candidates]


def save_inference_result(
    image_path: Path,
    input_path: Path,
    annotations: Sequence[dict[str, Any]],
    output_json_path: Path,
) -> None:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_file": resolve_image_file_field(image_path, input_path),
        "annotation": list(annotations),
    }
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_inference_for_single_image(
    image_path: Path,
    input_path: Path,
    output_dir: Path,
    model,
    processor: Sam3Processor,
    points_per_side: int,
    points_batch_size: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    stability_score_offset: float,
    min_mask_area: int,
    mask_nms_thresh: float,
) -> Path:
    with Image.open(image_path) as pil_image:
        image = pil_image.convert("RGB")

    annotations = generate_automatic_masks_for_single_image(
        model=model,
        processor=processor,
        image=image,
        points_per_side=points_per_side,
        points_batch_size=points_batch_size,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        min_mask_area=min_mask_area,
        mask_nms_thresh=mask_nms_thresh,
    )

    output_json_path = resolve_output_json_path(image_path, input_path, output_dir)
    save_inference_result(
        image_path=image_path,
        input_path=input_path,
        annotations=annotations,
        output_json_path=output_json_path,
    )
    return output_json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch SAM3 automatic mask inference without class prompts."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Single image path or an image directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to store automatic mask json files.",
    )
    parser.add_argument(
        "--hf-ckpt-path",
        type=Path,
        required=True,
        help="Local HuggingFace SAM3 checkpoint directory or checkpoint file path.",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=DEFAULT_POINTS_PER_SIDE,
        help="How many grid points are sampled on each image side.",
    )
    parser.add_argument(
        "--points-batch-size",
        type=int,
        default=DEFAULT_POINTS_BATCH_SIZE,
        help="How many point prompts are inferred in one predictor batch.",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=DEFAULT_PRED_IOU_THRESH,
        help="Predicted IoU threshold used to filter candidate masks.",
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=DEFAULT_STABILITY_SCORE_THRESH,
        help="Stability score threshold used to filter candidate masks.",
    )
    parser.add_argument(
        "--stability-score-offset",
        type=float,
        default=DEFAULT_STABILITY_SCORE_OFFSET,
        help="Delta used when computing stability scores from mask logits.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=DEFAULT_MIN_MASK_AREA,
        help="Discard masks whose pixel area is smaller than this threshold.",
    )
    parser.add_argument(
        "--mask-nms-thresh",
        type=float,
        default=DEFAULT_MASK_NMS_THRESH,
        help="Mask IoU threshold used for NMS deduplication.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=("cpu", "cuda"),
        help="Inference device.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch compile when building SAM3.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.points_per_side <= 0:
        raise ValueError("--points-per-side must be a positive integer")
    if args.points_batch_size <= 0:
        raise ValueError("--points-batch-size must be a positive integer")
    if args.min_mask_area < 0:
        raise ValueError("--min-mask-area must be non-negative")
    if not 0.0 <= args.pred_iou_thresh <= 1.0:
        raise ValueError("--pred-iou-thresh must be between 0 and 1")
    if not 0.0 <= args.stability_score_thresh <= 1.0:
        raise ValueError("--stability-score-thresh must be between 0 and 1")
    if args.stability_score_offset < 0.0:
        raise ValueError("--stability-score-offset must be non-negative")
    if not 0.0 <= args.mask_nms_thresh <= 1.0:
        raise ValueError("--mask-nms-thresh must be between 0 and 1")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device=cuda was requested")

    resolve_hf_checkpoint_path(args.hf_ckpt_path)


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = torch.device(args.device)
    with build_inference_context(device):
        model = build_model(
            device=device,
            hf_ckpt_path=args.hf_ckpt_path,
            compile_model=args.compile_model,
        )
        if model.inst_interactive_predictor is None:
            raise RuntimeError("SAM3 interactive image predictor was not initialized")
        processor = Sam3Processor(model=model, device=str(device))

        image_paths = resolve_image_paths(args.input_path)
        if not image_paths:
            raise FileNotFoundError(f"No images found under: {args.input_path}")

        progress_bar = tqdm(
            image_paths,
            desc="SAM3 auto masks",
            unit="image",
        )
        for image_path in progress_bar:
            run_inference_for_single_image(
                image_path=image_path,
                input_path=args.input_path,
                output_dir=args.output_dir,
                model=model,
                processor=processor,
                points_per_side=args.points_per_side,
                points_batch_size=args.points_batch_size,
                pred_iou_thresh=args.pred_iou_thresh,
                stability_score_thresh=args.stability_score_thresh,
                stability_score_offset=args.stability_score_offset,
                min_mask_area=args.min_mask_area,
                mask_nms_thresh=args.mask_nms_thresh,
            )


if __name__ == "__main__":
    main()
