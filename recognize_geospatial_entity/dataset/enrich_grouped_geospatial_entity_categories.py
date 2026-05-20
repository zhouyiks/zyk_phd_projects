from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "merged_geospatial_entity_category_grouped_cn.json"
OUTPUT_PATH = INPUT_PATH


SRC_NBS_CN = {
    "名称": "国家统计局《主要统计指标解释》",
    "链接": "https://www.stats.gov.cn/sj/ndsj/2021/html/zb08.htm",
}

SRC_NBS_EN = {
    "名称": "National Bureau of Statistics of China, Explanatory Notes on Main Statistical Indicators",
    "链接": "https://www.stats.gov.cn/sj/ndsj/2021/html/zbe08.htm",
}

SRC_GB50137 = {
    "名称": "GB 50137-2011《城市用地分类与规划建设用地标准》",
    "链接": "https://www.yunfu.gov.cn/attachment/551a019054f111e99ab7fa163ec567b2/1/%E5%9F%8E%E5%B8%82%E7%94%A8%E5%9C%B0%E5%88%86%E7%B1%BB%E4%B8%8E%E8%A7%84%E5%88%92%E5%BB%BA%E8%AE%BE%E7%94%A8%E5%9C%B0%E6%A0%87%E5%87%86.pdf",
}

SRC_GZ_TRANSPORT = {
    "名称": "广州市规划和自然资源局《交通运输用地地类认定标准？》",
    "链接": "https://ghzyj.gz.gov.cn/sjb/fw/cjwt/content/post_10157496.html",
}

SRC_GZ_WATER = {
    "名称": "广州市规划和自然资源局《水域及水利设施用地地类认定标准？》",
    "链接": "https://ghzyj.gz.gov.cn/hdjl/ywzsk/zygl/content/mpost_10157492.html",
}


OFFICIAL_MATCHES = {
    "cultivated_land": {
        "标准类别中文": "耕地",
        "标准类别英文": "Cultivated Land",
        "官方定义": (
            "指利用地表耕作层种植农作物为主，每年种植一季及以上（含以一年一季以上的耕种方式"
            "种植多年生作物）的土地，包括熟地、新开发、复垦、整理地、休闲地（含轮歇地、休耕"
            "地）；以及间有零星果树、桑树或其他树木的耕地；包括南方宽度＜1.0米、北方宽度＜"
            "2.0米固定的沟、渠、路和地坎（埂）；包括直接利用地表耕作层种植的温室、大棚、地"
            "膜等保温、保湿设施用地。"
        ),
        "来源": [SRC_NBS_CN, SRC_NBS_EN],
    },
    "garden_land": {
        "标准类别中文": "园地",
        "标准类别英文": "Garden Land",
        "官方定义": (
            "指种植以采集果、叶、根、茎、枝、汁等为主的集约经营的多年生木本和草本作物，覆盖"
            "度大于50%和每亩株数大于合理株数70%的土地。包括用于育苗的土地。"
        ),
        "来源": [SRC_NBS_CN, SRC_NBS_EN],
    },
    "forest_land": {
        "标准类别中文": "林地",
        "标准类别英文": "Forest Land",
        "官方定义": (
            "指生长乔木、竹类、灌木的土地。不包括生长林木的湿地，城镇、村庄范围内的绿化林木"
            "用地，铁路、公路征地范围内的林木，以及河流、沟渠的护堤林用地。"
        ),
        "来源": [SRC_NBS_CN, SRC_NBS_EN],
    },
    "grassland": {
        "标准类别中文": "草地",
        "标准类别英文": "Grassland",
        "官方定义": (
            "指生长草本植物为主的土地，包括乔木郁闭度＜0.1的疏林草地、灌木覆盖度＜40％的灌"
            "丛草地，不包括生长草本植物的湿地。"
        ),
        "来源": [SRC_NBS_CN, SRC_NBS_EN],
    },
    "wetland": {
        "标准类别中文": "湿地",
        "标准类别英文": "Wetland",
        "官方定义": (
            "指陆地和水域的交汇处，水位接近或处于地表面，或有浅层积水，且处于自然状态的土"
            "地。"
        ),
        "来源": [SRC_NBS_CN, SRC_NBS_EN],
    },
    "urban_rural_industrial_mining": {
        "标准类别中文": "城镇村及工矿用地",
        "标准类别英文": "Land for Urban, Rural, Industrial and Mining Activities",
        "官方定义": (
            "指城乡居民点、独立居民点以及居民点以外的工矿、国防、名胜古迹等企事业单位用地，"
            "包括其内部交通、绿化用地。"
        ),
        "来源": [SRC_NBS_CN, SRC_NBS_EN],
    },
    "transport_land": {
        "标准类别中文": "交通运输用地",
        "标准类别英文": "Land Used for Transport",
        "官方定义": (
            "指用于运输通行的地面线路、场站等的土地。包括民用机场、汽车客货运场站、港口、码"
            "头、地面运输管道和各种道路以及轨道交通用地。"
        ),
        "来源": [SRC_NBS_CN, SRC_NBS_EN, SRC_GZ_TRANSPORT],
    },
    "water_and_water_conservancy": {
        "标准类别中文": "水域及水利设施用地",
        "标准类别英文": "Land Used for Water and Water Conservancy Facilities",
        "官方定义": (
            "指陆地水域、沟渠、水工建筑物等用地。不包括滞洪区。"
        ),
        "来源": [SRC_NBS_CN, SRC_NBS_EN, SRC_GZ_WATER],
    },
    "urban_development_land": {
        "标准类别中文": "城市建设用地",
        "标准类别英文": "urban development land",
        "官方定义": (
            "指城市和县人民政府所在地镇内的居住用地、公共管理与公共服务用地、商业服务业设施"
            "用地、工业用地、物流仓储用地、交通设施用地、公用设施用地、绿地。"
        ),
        "来源": [SRC_GB50137],
    },
    "agriforest_land": {
        "标准类别中文": "农林用地",
        "标准类别英文": None,
        "官方定义": (
            "涵盖农林业生产的用地，包括《土地利用现状分类》一级地类“耕地”“园地”“林地”"
            "与二级地类“天然牧草地”“人工牧草地”“设施农用地”“田坎”“农村道路”，对应"
            "农用地除去“坑塘水面”和“沟渠”的地类。"
        ),
        "来源": [SRC_GB50137],
    },
    "other_nondevelopment_land": {
        "标准类别中文": "其他非建设用地",
        "标准类别英文": None,
        "官方定义": (
            "空闲地、盐碱地、沼泽地、沙地、裸地、不用于畜牧业的草地等用地。"
        ),
        "来源": [SRC_GB50137],
    },
    "residential": {
        "标准类别中文": "居住用地",
        "标准类别英文": "residential",
        "官方定义": "住宅和相应服务设施的用地。",
        "来源": [SRC_GB50137],
    },
    "public_admin_services": {
        "标准类别中文": "公共管理与公共服务用地",
        "标准类别英文": "administration and public services",
        "官方定义": (
            "行政、文化、教育、体育、卫生等机构和设施的用地，不包括居住用地中的服务设施用"
            "地。"
        ),
        "来源": [SRC_GB50137],
    },
    "historic_sites_land": {
        "标准类别中文": "文物古迹用地",
        "标准类别英文": None,
        "官方定义": (
            "具有历史、艺术、科学价值且没有其他使用功能的建筑物、构筑物、遗址、墓葬等用地。"
        ),
        "来源": [SRC_GB50137],
    },
    "sports_land": {
        "标准类别中文": "体育用地",
        "标准类别英文": None,
        "官方定义": (
            "体育场馆和体育训练基地等用地，不包括学校等机构专用的体育设施用地。"
        ),
        "来源": [SRC_GB50137],
    },
    "commercial_business": {
        "标准类别中文": "商业服务业设施用地",
        "标准类别英文": "commercial and business facilities",
        "官方定义": (
            "指各类商业、商务、娱乐康体等设施用地，其核心内涵是以营利为主要目的商业服务设"
            "施。"
        ),
        "来源": [SRC_GB50137],
    },
    "green_space": {
        "标准类别中文": "绿地",
        "标准类别英文": "green space",
        "官方定义": (
            "公园绿地、防护绿地等开放空间用地，不包括住区、单位内部配建的绿地。"
        ),
        "来源": [SRC_GB50137],
    },
    "industrial": {
        "标准类别中文": "工业用地",
        "标准类别英文": "industrial",
        "官方定义": (
            "工矿企业的生产车间、库房及其附属设施等用地，包括专用的铁路、码头和道路等用地，"
            "不包括露天矿用地。"
        ),
        "来源": [SRC_GB50137],
    },
    "logistics_warehouse": {
        "标准类别中文": "物流仓储用地",
        "标准类别英文": "logistics and warehouse",
        "官方定义": (
            "物资储备、中转、配送、批发、交易等的用地，包括大型批发市场以及货运公司车队的站"
            "场（不包括加工）等用地。"
        ),
        "来源": [SRC_GB50137],
    },
    "utilities": {
        "标准类别中文": "公用设施用地",
        "标准类别英文": "municipal utilities",
        "官方定义": (
            "供应、环境、安全等设施用地。"
        ),
        "来源": [SRC_GB50137],
    },
    "other_utilities": {
        "标准类别中文": "其他公用设施用地",
        "标准类别英文": None,
        "官方定义": (
            "除以上之外的公用设施用地，包括施工、养护、维修设施等用地。"
        ),
        "来源": [SRC_GB50137],
    },
    "regional_utilities": {
        "标准类别中文": "区域公用设施用地",
        "标准类别英文": None,
        "官方定义": (
            "为区域服务的公用设施用地，包括区域性能源设施、水工设施、通讯设施、殡葬设施、环"
            "卫设施、排水设施等用地。"
        ),
        "来源": [SRC_GB50137],
    },
    "mining_land": {
        "标准类别中文": "采矿用地",
        "标准类别英文": None,
        "官方定义": (
            "采矿、采石、采沙、盐田、砖瓦窑等地面生产用地及尾矿堆放地。"
        ),
        "来源": [SRC_GB50137],
    },
    "military_land": {
        "标准类别中文": "军事用地",
        "标准类别英文": None,
        "官方定义": (
            "专门用于军事目的的设施用地，不包括部队家属生活区和军民共用设施等用地。"
        ),
        "来源": [SRC_GB50137],
    },
    "security_land": {
        "标准类别中文": "安保用地",
        "标准类别英文": None,
        "官方定义": (
            "监狱、拘留所、劳改场所和安全保卫设施等用地，不包括公安局用地。"
        ),
        "来源": [SRC_GB50137],
    },
}


GROUP_METADATA = {
    "耕地": {
        "公共类别名英文": "Cultivated Land",
        "official_key": "cultivated_land",
        "匹配方式": "exact",
        "备注": "当前公共类别与国家统计局土地利用统计口径直接对应。",
    },
    "园地": {
        "公共类别名英文": "Garden Land",
        "official_key": "garden_land",
        "匹配方式": "exact",
        "备注": "当前公共类别与国家统计局土地利用统计口径直接对应。",
    },
    "设施农业与养殖用地": {
        "公共类别名英文": "Facility Agriculture and Aquaculture Land",
        "official_key": "agriforest_land",
        "匹配方式": "approximate",
        "备注": (
            "当前分组是项目自定义归并类，近似参考“农林用地/设施农用地”口径；其中 aquaculture、"
            "greenhouse 等更接近设施农用地，barn 更偏农业生产配套设施。"
        ),
    },
    "林地": {
        "公共类别名英文": "Forest Land",
        "official_key": "forest_land",
        "匹配方式": "exact",
        "备注": "当前公共类别与国家统计局土地利用统计口径直接对应。",
    },
    "灌丛地": {
        "公共类别名英文": "Shrubland",
        "official_key": "forest_land",
        "匹配方式": "approximate",
        "备注": "当前分组以灌丛、灌木覆盖地为主，近似归入官方“林地”口径中的灌木地带。",
    },
    "草地": {
        "公共类别名英文": "Grassland",
        "official_key": "grassland",
        "匹配方式": "exact",
        "备注": "当前公共类别与国家统计局土地利用统计口径直接对应。",
    },
    "裸土地表": {
        "公共类别名英文": "Bare Surface",
        "official_key": "other_nondevelopment_land",
        "匹配方式": "approximate",
        "备注": "当前分组覆盖裸地、沙地、荒漠和岩石地表等，近似参考官方“其他非建设用地”。",
    },
    "海岸与岸滩": {
        "公共类别名英文": "Coastal and Shore Land",
        "official_key": "water_and_water_conservancy",
        "匹配方式": "approximate",
        "备注": (
            "当前分组主要对应官方口径中的沿海滩涂、内陆滩涂及自然水域边缘带；项目分组比官方大"
            "类更偏自然岸线与岸滩景观。"
        ),
    },
    "冰雪地物": {
        "公共类别名英文": "Ice and Snow Features",
        "official_key": "water_and_water_conservancy",
        "匹配方式": "approximate",
        "备注": "当前分组主要对应官方水域及水利设施用地中的“冰川及永久积雪”等自然冰雪地物。",
    },
    "内陆与海洋水域": {
        "公共类别名英文": "Inland and Marine Waters",
        "official_key": "water_and_water_conservancy",
        "匹配方式": "approximate",
        "备注": (
            "当前分组以自然或人工水域为主，并合并了 wetland；官方标准中“湿地”可单独成类，因"
            "此此处分组范围略宽于单一官方类别。"
        ),
    },
    "人工池体": {
        "公共类别名英文": "Artificial Water Pools",
        "official_key": "water_and_water_conservancy",
        "匹配方式": "approximate",
        "备注": "当前分组更接近官方口径中的坑塘水面、沟渠及其他人工小型蓄水水域。",
    },
    "山地地貌": {
        "公共类别名英文": "Mountain Landform",
        "official_key": "other_nondevelopment_land",
        "匹配方式": "approximate",
        "备注": "当前分组属于地貌对象类，官方土地利用标准中通常不单独以“山地地貌”成类。",
    },
    "居住建筑": {
        "公共类别名英文": "Residential Buildings",
        "official_key": "residential",
        "匹配方式": "approximate",
        "备注": "当前分组以居住建筑物为对象，近似对应官方“居住用地”。",
    },
    "商业服务设施": {
        "公共类别名英文": "Commercial Service Facilities",
        "official_key": "commercial_business",
        "匹配方式": "exact",
        "备注": "当前分组与城乡规划标准中的商业服务业设施用地高度对应。",
    },
    "公共服务设施": {
        "公共类别名英文": "Public Service Facilities",
        "official_key": "public_admin_services",
        "匹配方式": "exact",
        "备注": "当前分组与城乡规划标准中的公共管理与公共服务用地高度对应。",
    },
    "文化与历史景观": {
        "公共类别名英文": "Cultural and Historic Landscapes",
        "official_key": "historic_sites_land",
        "匹配方式": "approximate",
        "备注": "当前分组主要近似参考“文物古迹用地”；其中 palace 等文化景观对象在项目分组里被一并纳入。",
    },
    "休闲游憩设施": {
        "公共类别名英文": "Leisure and Recreation Facilities",
        "official_key": "green_space",
        "匹配方式": "approximate",
        "备注": (
            "当前分组混合了公园广场型开放空间与部分游憩设施，近似参考官方“绿地”口径；其中 "
            "amusement park 等也与“娱乐康体用地”相关。"
        ),
    },
    "体育场地": {
        "公共类别名英文": "Sports Venues",
        "official_key": "sports_land",
        "匹配方式": "exact",
        "备注": "当前分组与城乡规划标准中的体育用地高度对应。",
    },
    "普通建筑与构筑物": {
        "公共类别名英文": "General Buildings and Structures",
        "official_key": "urban_rural_industrial_mining",
        "匹配方式": "approximate",
        "备注": "当前分组是泛化建筑目标类，更接近建设用地或城镇村及工矿用地中的一般建筑实体。",
    },
    "人工硬化地表": {
        "公共类别名英文": "Impervious Surface",
        "official_key": "urban_rural_industrial_mining",
        "匹配方式": "approximate",
        "备注": "当前分组是表面属性类，近似对应建设用地中已开发硬化地表，但并非官方单独地类名称。",
    },
    "透水地表": {
        "公共类别名英文": "Pervious Surface",
        "official_key": None,
        "匹配方式": "none",
        "备注": "当前分组属于表面渗透性属性类，未检索到与土地利用/城乡用地官方标准直接同名对应项。",
    },
    "工业用地与设施": {
        "公共类别名英文": "Industrial Land and Facilities",
        "official_key": "industrial",
        "匹配方式": "exact",
        "备注": "当前分组与城乡规划标准中的工业用地高度对应。",
    },
    "仓储设施": {
        "公共类别名英文": "Storage and Tank Facilities",
        "official_key": "logistics_warehouse",
        "匹配方式": "approximate",
        "备注": "当前分组以仓储、集装箱和储罐设施为主，近似参考官方“物流仓储用地”。",
    },
    "工程施工场地": {
        "公共类别名英文": "Construction Sites",
        "official_key": "other_utilities",
        "匹配方式": "approximate",
        "备注": "construction site 与 digging pile 更接近施工、养护、维修设施相关的阶段性建设场地。",
    },
    "采矿用地": {
        "公共类别名英文": "Mining Land",
        "official_key": "mining_land",
        "匹配方式": "exact",
        "备注": "当前分组与城乡规划标准中的采矿用地直接对应。",
    },
    "能源电力设施": {
        "公共类别名英文": "Energy and Power Facilities",
        "official_key": "utilities",
        "匹配方式": "approximate",
        "备注": "当前分组以变电、发电、输电等设施为主，近似参考官方“公用设施用地”。",
    },
    "市政公用设施": {
        "公共类别名英文": "Municipal Utilities",
        "official_key": "regional_utilities",
        "匹配方式": "approximate",
        "备注": "当前分组混合了水工、排污、输送和能源营业网点设施，近似参考区域公用设施用地。",
    },
    "军事与航天设施": {
        "公共类别名英文": "Military and Aerospace Facilities",
        "official_key": "military_land",
        "匹配方式": "approximate",
        "备注": "military facility 可直接参考军事用地；space facility 属项目扩展目标类，因此整体按近似匹配处理。",
    },
    "机场设施": {
        "公共类别名英文": "Airport Facilities",
        "official_key": "transport_land",
        "匹配方式": "approximate",
        "备注": "当前分组主要对应交通运输用地中的机场用地及其附属设施。",
    },
    "航空器": {
        "公共类别名英文": "Aircraft",
        "official_key": None,
        "匹配方式": "none",
        "备注": "当前分组属于移动目标类，未检索到与土地利用/城乡用地官方标准直接同名对应项。",
    },
    "道路": {
        "公共类别名英文": "Roads",
        "official_key": "transport_land",
        "匹配方式": "approximate",
        "备注": "当前分组主要对应交通运输用地中的公路、城镇村道路等子类。",
    },
    "道路交叉设施": {
        "公共类别名英文": "Road Intersections",
        "official_key": "transport_land",
        "匹配方式": "approximate",
        "备注": "当前分组主要对应道路及其交叉口等交通节点设施。",
    },
    "桥梁设施": {
        "公共类别名英文": "Bridges",
        "official_key": "transport_land",
        "匹配方式": "approximate",
        "备注": "当前分组主要对应交通运输用地中服务道路、铁路通行的桥梁及附属设施。",
    },
    "交通附属设施": {
        "公共类别名英文": "Transport Ancillary Facilities",
        "official_key": "transport_land",
        "匹配方式": "approximate",
        "备注": "当前分组主要对应交通服务场站、停车场和交通附属设施等。",
    },
    "铁路设施": {
        "公共类别名英文": "Railway Facilities",
        "official_key": "transport_land",
        "匹配方式": "approximate",
        "备注": "当前分组主要对应交通运输用地中的铁路用地及场站设施。",
    },
    "港口与航运设施": {
        "公共类别名英文": "Port and Navigation Facilities",
        "official_key": "transport_land",
        "匹配方式": "approximate",
        "备注": "当前分组主要对应交通运输用地中的港口码头用地及其附属设施。",
    },
    "船舶": {
        "公共类别名英文": "Vessels",
        "official_key": None,
        "匹配方式": "none",
        "备注": "当前分组属于移动目标类，未检索到与土地利用/城乡用地官方标准直接同名对应项。",
    },
    "机动车": {
        "公共类别名英文": "Motor Vehicles",
        "official_key": None,
        "匹配方式": "none",
        "备注": "当前分组属于移动目标类，未检索到与土地利用/城乡用地官方标准直接同名对应项。",
    },
    "工程与特种车辆": {
        "公共类别名英文": "Engineering and Special Vehicles",
        "official_key": None,
        "匹配方式": "none",
        "备注": "当前分组属于移动目标类，未检索到与土地利用/城乡用地官方标准直接同名对应项。",
    },
    "军事车辆": {
        "公共类别名英文": "Military Vehicles",
        "official_key": None,
        "匹配方式": "none",
        "备注": "当前分组属于移动目标类，未检索到与土地利用/城乡用地官方标准直接同名对应项。",
    },
    "云层": {
        "公共类别名英文": "Clouds",
        "official_key": None,
        "匹配方式": "none",
        "备注": "当前分组属于大气目标类，未检索到与土地利用/城乡用地官方标准直接同名对应项。",
    },
}


def load_grouped_categories() -> list[tuple[str, list[str]]]:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    grouped: list[tuple[str, list[str]]] = []
    if not isinstance(data, list):
        raise ValueError(f"Invalid grouped json format: {INPUT_PATH}")

    for item in data:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Each item must be a single-key dict: {item}")
        group_name, members = next(iter(item.items()))
        if not isinstance(group_name, str) or not isinstance(members, list):
            raise ValueError(f"Invalid group entry: {item}")
        grouped.append((group_name, members))

    return grouped


def build_enriched_output(grouped: list[tuple[str, list[str]]]) -> list[dict]:
    output: list[dict] = []
    seen = set()

    for group_name, members in grouped:
        if group_name in seen:
            raise ValueError(f"Duplicated group name: {group_name}")
        seen.add(group_name)

        meta = GROUP_METADATA.get(group_name)
        if meta is None:
            raise ValueError(f"Missing metadata for group: {group_name}")

        official_key = meta["official_key"]
        official_match = {
            "匹配方式": meta["匹配方式"],
            "标准类别中文": None,
            "标准类别英文": None,
            "官方定义": None,
            "来源": [],
        }
        if official_key is not None:
            official = deepcopy(OFFICIAL_MATCHES[official_key])
            official_match.update(official)

        output.append(
            {
                "公共类别名": group_name,
                "公共类别名英文": meta["公共类别名英文"],
                "官方标准匹配": official_match,
                "备注": meta["备注"],
                "包含类别": members,
            }
        )

    if set(GROUP_METADATA) != {name for name, _ in grouped}:
        missing_in_json = sorted(set(GROUP_METADATA) - {name for name, _ in grouped})
        extra_in_json = sorted({name for name, _ in grouped} - set(GROUP_METADATA))
        raise ValueError(
            f"Metadata/group mismatch. missing_in_json={missing_in_json}, "
            f"extra_in_json={extra_in_json}"
        )

    return output


def main() -> None:
    grouped = load_grouped_categories()
    output = build_enriched_output(grouped)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved enriched grouped categories to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
