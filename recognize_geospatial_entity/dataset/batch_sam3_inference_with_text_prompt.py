from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from itertools import count
from pathlib import Path
from typing import Iterator, Sequence

import torch
from PIL import Image
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = PROJECT_ROOT / "sam3_main"

if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

import sam3
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)


"""
目标：给定固定的类别名称列表，包含23种类别，实现批量文本提示 SAM3 推理。

实现说明：
1. 按照 sam3_image_batched_inference.ipynb 的 Datapoint -> transform -> collate ->
   model -> postprocess 流程组织代码。
2. 默认使用现有的两个类别 JSON 合并生成
   `candidate_geospatial_entity_category_list`。
3. 每次取 BATCH_SIZE 个类别构建文本提示，对单张图像做一次批量推理。
4. 每张图像输出一个同名 json 文件，内容格式为：
   {
     "image_file": "relative/path/to/image_name.jpg",
     "annotation": [
       {"class_name": "category_name#1", "segmentation": [rle#1, rle#2, ...]},
       {"class_name": "category_name#2", "segmentation": [rle#1, rle#2, ...]}
     ]
   }

注意：
1. 当前脚本只完成流程框架搭建，不在此脚本内做验证或执行。
2. 真正运行前，需要提供用户本地已下载的 HuggingFace SAM3 权重目录或文件路径。
"""


DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = 1008
DEFAULT_DETECTION_THRESHOLD = 0.4
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sam3_batch_text_prompt_outputs"
DEFAULT_CATEGORY_FILE = PROJECT_ROOT / "sam3_geospatial_entity_category.json"
DEFAULT_CATEGORY_FILES = (DEFAULT_CATEGORY_FILE,)
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


def normalize_category_name(name: str) -> str:
    return " ".join(name.casefold().replace("_", " ").replace("-", " ").split())


def load_category_names(category_file: Path) -> list[str]:
    if not category_file.exists():
        raise FileNotFoundError(f"Category file does not exist: {category_file}")

    with category_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Category file must contain a JSON list: {category_file}")

    categories = []
    for item in data:
        if not isinstance(item, str):
            continue
        normalized = normalize_category_name(item)
        if normalized:
            categories.append(normalized)
    return categories


def build_candidate_geospatial_entity_category_list(
    category_files: Sequence[Path] = DEFAULT_CATEGORY_FILES,
) -> list[str]:
    categories = set()
    for category_file in category_files:
        categories.update(load_category_names(category_file))
    return sorted(categories)


try:
    candidate_geospatial_entity_category_list = (
        build_candidate_geospatial_entity_category_list()
    )
except FileNotFoundError:
    candidate_geospatial_entity_category_list = []


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


def chunked(items: Sequence[str], chunk_size: int) -> Iterator[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def resolve_bpe_path() -> str:
    return str(Path(sam3.__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")


def build_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> ComposeAPI:
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=image_size,
                max_size=image_size,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_postprocessor(
    device: torch.device,
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
) -> PostProcessImage:
    return PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=True,
        always_interpolate_masks_on_gpu=device.type == "cuda",
        detection_threshold=detection_threshold,
        to_cpu=True,
    )


def build_inference_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


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
    )


def create_empty_datapoint() -> Datapoint:
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint: Datapoint, pil_image: Image.Image) -> None:
    width, height = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=(height, width))]


def add_text_prompt(
    datapoint: Datapoint,
    text_query: str,
    query_id: int,
) -> int:
    if len(datapoint.images) != 1:
        raise ValueError("Please set exactly one image before adding text prompts")

    height, width = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=(height, width),
                object_id=0,
                frame_index=0,
            ),
        )
    )
    return query_id


def build_text_prompt(class_name: str, text_prompt_template: str) -> str:
    # 先保留为模板钩子，后续如果需要更强约束，可在这里统一改写 prompt。
    return text_prompt_template.format(class_name=class_name)


def serialize_rle_list(rle_list: Sequence[dict] | None) -> list[dict]:
    serialized_rles = []
    for rle in rle_list or []:
        if not isinstance(rle, dict):
            continue

        serialized = dict(rle)
        counts = serialized.get("counts")
        size = serialized.get("size")

        if isinstance(counts, bytes):
            serialized["counts"] = counts.decode("utf-8")
        if isinstance(size, tuple):
            serialized["size"] = list(size)

        serialized_rles.append(serialized)
    return serialized_rles


def infer_text_prompt_batch_for_single_image(
    image: Image.Image,
    class_names: Sequence[str],
    model,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    device: torch.device,
    query_id_counter,
) -> list[dict]:
    datapoint = create_empty_datapoint()
    set_image(datapoint, image.copy())

    query_id_to_class_name: dict[int, str] = {}
    for class_name in class_names:
        query_id = next(query_id_counter)
        query_text = class_name # build_text_prompt(class_name, text_prompt_template)
        add_text_prompt(datapoint, query_text, query_id)
        query_id_to_class_name[query_id] = class_name

    datapoint = transform(datapoint)
    batch = collate([datapoint], dict_key="dummy")["dummy"]
    batch = copy_data_to_device(
        batch,
        device,
        non_blocking=device.type == "cuda",
    )

    # with torch.inference_mode():
    with torch.no_grad():
        output = model(batch)

    processed_results = postprocessor.process_results(output, batch.find_metadatas)

    annotations = []
    for query_id, class_name in query_id_to_class_name.items():
        result = processed_results.get(query_id, {})
        masks_rle = serialize_rle_list(result.get("masks_rle"))
        annotations.append(
            {
                "class_name": class_name,
                "segmentation": masks_rle,
            }
        )
    return annotations


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


def save_inference_result(
    image_path: Path,
    input_path: Path,
    annotations: Sequence[dict],
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
    class_names: Sequence[str],
    batch_size: int,
    model,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    device: torch.device,
    query_id_counter,
) -> Path:
    with Image.open(image_path) as pil_image:
        image = pil_image.convert("RGB")

    all_annotations = []
    for class_name_batch in chunked(class_names, batch_size):
        batch_annotations = infer_text_prompt_batch_for_single_image(
            image=image,
            class_names=class_name_batch,
            model=model,
            transform=transform,
            postprocessor=postprocessor,
            device=device,
            query_id_counter=query_id_counter,
        )
        all_annotations.extend(batch_annotations)

    output_json_path = resolve_output_json_path(image_path, input_path, output_dir)
    save_inference_result(image_path, input_path, all_annotations, output_json_path)
    return output_json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch SAM3 inference with text prompts for geospatial categories."
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
        help="Directory used to store inference json files.",
    )
    parser.add_argument(
        "--category-files",
        type=Path,
        nargs="*",
        default=list(DEFAULT_CATEGORY_FILES),
        help="Category json files used to build candidate_geospatial_entity_category_list.",
    )
    parser.add_argument(
        "--hf-ckpt-path",
        type=Path,
        required=True,
        help="Local HuggingFace SAM3 checkpoint directory or checkpoint file path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="How many category names are packed into one SAM3 inference batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=("cpu", "cuda"),
        help="Inference device.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square resize size used by the SAM3 preprocessing pipeline.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=DEFAULT_DETECTION_THRESHOLD,
        help="Detection score threshold used in SAM3 post-processing.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch compile when building SAM3.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device=cuda was requested")

    resolve_hf_checkpoint_path(args.hf_ckpt_path)


def main() -> None:
    args = parse_args()
    validate_args(args)

    class_names = build_candidate_geospatial_entity_category_list(args.category_files)
    if not class_names:
        raise ValueError("No candidate geospatial entity categories were loaded")

    device = torch.device(args.device)
    with build_inference_context(device):
        model = build_model(
            device=device,
            hf_ckpt_path=args.hf_ckpt_path,
            compile_model=args.compile_model,
        )
        transform = build_transform(args.image_size)
        postprocessor = build_postprocessor(
            device=device,
            detection_threshold=args.detection_threshold,
        )

        image_paths = resolve_image_paths(args.input_path)
        if not image_paths:
            raise FileNotFoundError(f"No images found under: {args.input_path}")

        query_id_counter = count(1)
        progress_bar = tqdm(
            image_paths,
            desc="SAM3 inference",
            unit="image",
        )
        for image_path in progress_bar:
            run_inference_for_single_image(
                image_path=image_path,
                input_path=args.input_path,
                output_dir=args.output_dir,
                class_names=class_names,
                batch_size=args.batch_size,
                model=model,
                transform=transform,
                postprocessor=postprocessor,
                device=device,
                query_id_counter=query_id_counter,
            )


if __name__ == "__main__":
    main()
