from __future__ import annotations

import json
from pathlib import Path


"""
目标：统计 /media/disk3/dataset/zipped_formatted_datasets 下所有 json 文件中的图像路径。

实现要求：
1. 遍历该目录下所有 .json 文件；
2. 读取每个 json 文件中的每个样本条目；
3. 若条目包含字符串类型的 image_file 字段，则将其原样加入结果列表；
4. 不对路径做任何修改，不去重，不规范化；
5. 最终结果写入当前项目根目录下的 rsdenseseg_image_files.json。
"""

DATASET_ROOT = Path("/media/disk3/dataset/zipped_formatted_datasets")
OUTPUT_PATH = Path(
    "/media/disk3/zhouyikang/zyk_phd_projects/recognize_geospatial_entity/"
    "rsdenseseg_image_files.json"
)


def iter_json_files(dataset_root: Path):
    for path in sorted(dataset_root.rglob("*.json")):
        if path.is_file():
            yield path


def collect_image_files(dataset_root: Path) -> list[str]:
    image_files: list[str] = []

    for json_file in iter_json_files(dataset_root):
        with json_file.open("r", encoding="utf-8") as f:
            json_data = json.load(f)

        if not isinstance(json_data, list):
            continue

        for item in json_data:
            if not isinstance(item, dict):
                continue

            image_file = item.get("image_file")
            if isinstance(image_file, str):
                image_files.append(image_file)

    return image_files


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {DATASET_ROOT}")

    image_files = collect_image_files(DATASET_ROOT)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(image_files, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(image_files)} image paths to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
