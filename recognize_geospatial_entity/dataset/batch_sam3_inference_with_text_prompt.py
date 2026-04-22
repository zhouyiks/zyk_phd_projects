from __future__ import annotations

import argparse
import json
import sys
from itertools import count
from pathlib import Path
from typing import Iterator, Sequence

import torch
from PIL import Image


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
目标：给定固定的类别名称列表，预计包含300+种类别，实现批量文本提示 SAM3 推理。

实现说明：
1. 按照 sam3_image_batched_inference.ipynb 的 Datapoint -> transform -> collate ->
   model -> postprocess 流程组织代码。
2. 默认使用现有的两个类别 JSON 合并生成
   `candidate_geospatial_entity_category_list`。
3. 每次取 BATCH_SIZE 个类别构建文本提示，对单张图像做一次批量推理。
4. 每张图像输出一个同名 json 文件，内容格式为：
   {
     "image_file": "image_name.jpg",
     "annotation": [
       {"class_name": "category_name#1", "segmentation": [rle#1, rle#2, ...]},
       {"class_name": "category_name#2", "segmentation": [rle#1, rle#2, ...]}
     ]
   }

注意：
1. 当前脚本只完成流程框架搭建，不在此脚本内做验证或执行。
2. 真正运行前，需要提供 checkpoint 或显式允许从 HuggingFace 下载权重。
"""


DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = 1008
DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sam3_batch_text_prompt_outputs"
DEFAULT_CATEGORY_FILES = (
    PROJECT_ROOT / "oes_geospatial_entity_category_en.json",
    PROJECT_ROOT / "rsdenseseg_geospatial_entity_category_en.json",
)
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


def build_model(
    device: torch.device,
    checkpoint_path: Path | None = None,
    load_from_hf: bool = False,
    compile_model: bool = False,
):
    return build_sam3_image_model(
        bpe_path=resolve_bpe_path(),
        device=str(device),
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        load_from_HF=load_from_hf,
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
    text_prompt_template: str,
) -> list[dict]:
    datapoint = create_empty_datapoint()
    set_image(datapoint, image.copy())

    query_id_to_class_name: dict[int, str] = {}
    for class_name in class_names:
        query_id = next(query_id_counter)
        query_text = build_text_prompt(class_name, text_prompt_template)
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


def save_inference_result(
    image_path: Path,
    annotations: Sequence[dict],
    output_json_path: Path,
) -> None:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_file": image_path.name,
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
    text_prompt_template: str,
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
            text_prompt_template=text_prompt_template,
        )
        all_annotations.extend(batch_annotations)

    output_json_path = resolve_output_json_path(image_path, input_path, output_dir)
    save_inference_result(image_path, all_annotations, output_json_path)
    return output_json_path

# dataloader
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
def get_wds_loader(train_dir):
    def byte_decode(x):
        return x.decode("utf-8")
    
    train_url = os.path.join(train_dir, "{pub11,rs3}-train-{0000..0031}.tar")

    def my_decoder(key, value):
        if key.endswith(".img_content"):
            assert isinstance(value, bytes)
            value = Image.open(io.BytesIO(value))
            value = preproc(value)
        elif key.endswith(".img_name") or key.endswith(".caption"):
            value = byte_decode(value)
        return value

    train_dataset = wds.WebDataset(train_url).decode(my_decoder)
    train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=1)

    return train_dataloader




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
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Local SAM3 checkpoint path.",
    )
    parser.add_argument(
        "--load-from-hf",
        action="store_true",
        help="Allow SAM3 to download pretrained weights from HuggingFace if no checkpoint is provided.",
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
        "--text-prompt-template",
        type=str,
        default="{class_name}",
        help="Prompt template. Example: 'a satellite image of {class_name}'.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch compile when building SAM3.",
    )
    parser.add_argument(
        "--rs5m",
        action="store_true",
        help="RS5M webdataset train dir",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device=cuda was requested")

    if args.checkpoint_path is not None and not args.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {args.checkpoint_path}"
        )

    if args.checkpoint_path is None and not args.load_from_hf:
        raise ValueError(
            "Please provide --checkpoint-path, or use --load-from-hf explicitly"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    class_names = build_candidate_geospatial_entity_category_list(args.category_files)
    if not class_names:
        raise ValueError("No candidate geospatial entity categories were loaded")

    device = torch.device(args.device)
    model = build_model(
        device=device,
        checkpoint_path=args.checkpoint_path,
        load_from_hf=args.load_from_hf,
        compile_model=args.compile_model,
    )
    transform = build_transform(args.image_size)
    postprocessor = build_postprocessor(
        device=device,
        detection_threshold=args.detection_threshold,
    )

    image_paths = list(iter_image_files(args.input_path))
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {args.input_path}")

    query_id_counter = count(1)
    for image_path in image_paths:
        output_json_path = run_inference_for_single_image(
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
            text_prompt_template=args.text_prompt_template,
        )
        print(f"Saved inference result to: {output_json_path}")


if __name__ == "__main__":
    main()
