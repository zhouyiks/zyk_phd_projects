from __future__ import annotations

import json
from pathlib import Path


"""
目标：统计/media/disk3/dataset/zipped_formatted_datasets下所有json文件中的遥感目标类别
实现路径：
1. json文件的解析方法：
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    all_cls_name = []
    for item in json_data:
        if 'annotation' in item:
            annotation = item['annotation']
        else:
            annotation = item['segms']
        for cls_item in annotation:
            if 'class_name' in cls_item:
                cls_name = cls_item['class_name']
            else:
                cls_name = cls_item['class']
            segmentation = cls_item['segmentation']
            if cls_name not in all_cls_name:
                all_cls_name.append(cls_name)
2. 所有类别名称小写，用空格替代下划线，并去除前后空格，最后去重，结果写入列表存在/media/disk3/zhouyikang/zyk_phd_projects/recognize_geospatial_entity/rsdenseseg_geospatial_entity_category_en.json
"""

DATASET_ROOT = Path("/media/disk3/dataset/zipped_formatted_datasets")
OUTPUT_PATH = Path(
    "/media/disk3/zhouyikang/zyk_phd_projects/recognize_geospatial_entity/"
    "rsdenseseg_geospatial_entity_category_en.json"
)


def iter_json_files(dataset_root: Path):
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".json":
            yield path


def normalize_category_name(name: str) -> str:
    return " ".join(
        name.casefold().replace("_", " ").replace("-", " ").split()
    )


def extract_categories_from_json(json_file: Path) -> set[str]:
    with json_file.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    if not isinstance(json_data, list):
        return set()

    categories = set()
    for item in json_data:
        if not isinstance(item, dict):
            continue

        annotation = item.get("annotation", item.get("segms"))
        if not isinstance(annotation, list):
            continue

        for cls_item in annotation:
            if not isinstance(cls_item, dict):
                continue

            cls_name = cls_item.get("class_name", cls_item.get("class"))
            _segmentation = cls_item.get("segmentation")
            if not isinstance(cls_name, str):
                continue

            normalized = normalize_category_name(cls_name)
            if normalized:
                categories.add(normalized)

    return categories


def collect_categories(dataset_root: Path) -> list[str]:
    categories = set()
    for json_file in iter_json_files(dataset_root):
        categories.update(extract_categories_from_json(json_file))
    return sorted(categories)


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {DATASET_ROOT}")

    categories = collect_categories(DATASET_ROOT)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(categories)} categories to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
