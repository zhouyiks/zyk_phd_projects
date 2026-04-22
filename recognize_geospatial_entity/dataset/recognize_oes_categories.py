from __future__ import annotations

import json
from pathlib import Path


# 目标：统计openearthsensing数据集中的遥感地物类别名称
# 实现路径：
# 1. 数据集图像路径：/media/disk5/dataset/openearthsensing/All_data
# 2. 每张图片的父目录名称即为地物类别，比如对于图片路径：
#    /media/disk5/dataset/openearthsensing/All_data/AID/Pond/pond_1.jpg
#    其父目录名称为Pond，即为地物类别名称
# 3. 遍历/media/disk5/dataset/openearthsensing/All_data下的所有图片，统计所有类别名称，
#    所有名称小写，去重，用列表组织起来，最后保存为文件：
#    /media/disk3/zhouyikang/zyk_phd_projects/recognize_geospatial_entity/oes_geospatial_entity_category_en.json

DATASET_ROOT = Path("/media/disk5/dataset/openearthsensing/All_data")
OUTPUT_PATH = Path(
    "/media/disk3/zhouyikang/zyk_phd_projects/recognize_geospatial_entity/"
    "oes_geospatial_entity_category_en.json"
)
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}


def iter_image_files(dataset_root: Path):
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def normalize_category_name(name: str) -> str:
    return name.casefold().replace("_", " ").strip()


def collect_categories(dataset_root: Path) -> list[str]:
    categories = {
        normalize_category_name(image_path.parent.name)
        for image_path in iter_image_files(dataset_root)
        if normalize_category_name(image_path.parent.name)
    }
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
