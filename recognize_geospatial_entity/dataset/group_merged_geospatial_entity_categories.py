from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "merged_geospatial_entity_category_en.json"
OUTPUT_PATH = PROJECT_ROOT / "merged_geospatial_entity_category_grouped_cn.json"


# 按遥感测绘场景中的常见二级地物/目标类型做归并。
# 列表中的值保留原始英文类别名，键使用中文公共类别名。
GROUPS: list[tuple[str, list[str]]] = [
    (
        "耕地",
        [
            "agriculture land",
            "circular farmland",
            "cropland",
            "dry field",
            "eroded farmland",
            "farmland",
            "field",
            "irrigated land",
            "paddy field",
            "plowed lands",
            "terraced field",
        ],
    ),
    (
        "园地",
        [
            "garden plot",
            "orchard",
            "vegetable plot",
            "vineyard",
        ],
    ),
    (
        "设施农业与养殖用地",
        [
            "aquaculture",
            "barn",
            "greenhouse",
        ],
    ),
    (
        "林地",
        [
            "arbor woodland",
            "coniferous",
            "deciduous",
            "forest",
            "river protection forest",
            "tree",
            "woodland",
        ],
    ),
    (
        "灌丛地",
        [
            "brushwood",
            "chaparral",
            "shrub land",
            "sparse shrub land",
        ],
    ),
    (
        "草地",
        [
            "grassland",
            "low vegetation",
            "meadow",
            "rangeland",
            "vegetation",
        ],
    ),
    (
        "裸土地表",
        [
            "bare land",
            "barren",
            "desert",
            "rock land",
            "sand",
            "tundra",
        ],
    ),
    (
        "海岸与岸滩",
        [
            "beach",
            "coast",
            "island",
            "lakeshore",
            "sandbeach",
        ],
    ),
    (
        "冰雪地物",
        [
            "ice land",
            "sea ice",
            "snowberg",
        ],
    ),
    (
        "内陆与海洋水域",
        [
            "lake",
            "pond",
            "river",
            "sea",
            "stream",
            "water",
            "wetland",
        ],
    ),
    (
        "人工池体",
        [
            "pool",
            "swimming pool",
        ],
    ),
    (
        "山地地貌",
        [
            "mountain",
        ],
    ),
    (
        "居住建筑",
        [
            "apartment",
            "detached house",
            "house",
            "mobile home",
            "multi-unit residential",
            "rural residential",
            "single-unit residential",
        ],
    ),
    (
        "商业服务设施",
        [
            "center",
            "commercial area",
            "commercial building",
            "commercial building block",
            "shopping mall",
        ],
    ),
    (
        "公共服务设施",
        [
            "border checkpoint",
            "church",
            "hospital",
            "prison",
        ],
    ),
    (
        "文化与历史景观",
        [
            "archaeological site",
            "cemetery",
            "palace",
        ],
    ),
    (
        "休闲游憩设施",
        [
            "amusement park",
            "fountain",
            "playground",
            "recreational facility",
            "square",
            "zoo",
        ],
    ),
    (
        "体育场地",
        [
            "baseball field",
            "basketball court",
            "court",
            "football field",
            "golf course",
            "ground track field",
            "soccer ball field",
            "sports court",
            "stadium",
            "tennis court",
        ],
    ),
    (
        "普通建筑与构筑物",
        [
            "building",
            "dense tall building",
            "small construction",
            "structure",
        ],
    ),
    (
        "人工硬化地表",
        [
            "developed space",
            "impervious surface",
            "pavement",
        ],
    ),
    (
        "透水地表",
        [
            "pervious surface",
        ],
    ),
    (
        "工业用地与设施",
        [
            "chimney",
            "industrial land",
            "refinery",
            "smokestack",
            "steelsmelter",
        ],
    ),
    (
        "仓储设施",
        [
            "container",
            "storage tank",
            "watertank",
        ],
    ),
    (
        "工程施工场地",
        [
            "construction site",
            "digging pile",
        ],
    ),
    (
        "采矿用地",
        [
            "mine",
            "quarry",
            "surface mine",
        ],
    ),
    (
        "能源电力设施",
        [
            "electric substation",
            "nuclear powerplant",
            "single transmission tower",
            "solar power plant",
            "substation",
            "thermal power station",
            "tower",
            "wind turbine",
        ],
    ),
    (
        "市政公用设施",
        [
            "dam",
            "gas station",
            "pipeline",
            "sewage plant",
        ],
    ),
    (
        "军事与航天设施",
        [
            "military facility",
            "space facility",
        ],
    ),
    (
        "机场设施",
        [
            "airport",
            "airport hangar",
            "airport terminal",
            "apron",
            "hardstand",
            "helipad",
            "runway",
        ],
    ),
    (
        "航空器",
        [
            "airplane",
            "helicopter",
        ],
    ),
    (
        "道路",
        [
            "avenue",
            "highway",
            "road",
        ],
    ),
    (
        "道路交叉设施",
        [
            "crossroads",
            "interchange",
            "intersection",
            "roundabout",
        ],
    ),
    (
        "桥梁设施",
        [
            "bridge",
            "footbridge",
            "overpass",
            "railway bridge",
            "road bridge",
            "viaduct",
        ],
    ),
    (
        "交通附属设施",
        [
            "expressway service area",
            "expressway toll station",
            "ground transportation station",
            "parking lot",
        ],
    ),
    (
        "铁路设施",
        [
            "railway",
            "railway station",
            "train station",
        ],
    ),
    (
        "港口与航运设施",
        [
            "container crane",
            "dock",
            "harbor",
            "lighthouse",
            "pier",
            "port",
            "shipyard",
        ],
    ),
    (
        "船舶",
        [
            "barge",
            "boat",
            "bulk carrier",
            "car carrier",
            "civil yacht",
            "container ship",
            "dry cargo ship",
            "engineering ship",
            "fishing boat",
            "megayacht",
            "motorboat",
            "passenger ship",
            "ship",
            "tank ship",
            "towing vessel",
            "tugboat",
            "warship",
        ],
    ),
    (
        "机动车",
        [
            "bus",
            "car",
            "large vehicle",
            "small vehicle",
            "van",
            "vehicle",
        ],
    ),
    (
        "工程与特种车辆",
        [
            "cargo truck",
            "dump truck",
            "excavator",
            "tractor",
            "trailer",
            "truck tractor",
        ],
    ),
    (
        "军事车辆",
        [
            "tank",
        ],
    ),
    (
        "云层",
        [
            "cloud",
        ],
    ),
]


def load_source_categories() -> list[str]:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"Invalid category list: {INPUT_PATH}")
    return data


def validate_groups(source_categories: list[str]) -> None:
    source_set = set(source_categories)

    grouped_items: list[str] = []
    for _group_name, items in GROUPS:
        grouped_items.extend(items)

    grouped_set = set(grouped_items)

    duplicates = sorted({item for item in grouped_items if grouped_items.count(item) > 1})
    missing = sorted(source_set - grouped_set)
    unexpected = sorted(grouped_set - source_set)

    if duplicates:
        raise ValueError(f"Duplicated categories in GROUPS: {duplicates}")
    if missing:
        raise ValueError(f"Missing categories in GROUPS: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected categories in GROUPS: {unexpected}")


def build_output() -> list[dict[str, list[str]]]:
    return [{group_name: items} for group_name, items in GROUPS]


def main() -> None:
    source_categories = load_source_categories()
    validate_groups(source_categories)

    output_data = build_output()
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output_data)} grouped categories to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
