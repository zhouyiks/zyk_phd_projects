import re


MASK_GENERATION_EarthReason = {
    "annotation_path": "./data/annotation_256x4_8tokens_v2/EarthReason_41955.json",
    "data_path": "",
}

MASK_GENERATION_LaSeRS = {
    "annotation_path": "./data/annotation_256x4_8tokens_v2/LaSeRS_10000.json",
    "data_path": "",
}

MASK_GENERATION_RefsegRS = {
    "annotation_path": "./data/annotation_256x4_8tokens_v2/RefsegRS_2172.json",
    "data_path": "",
}

MASK_GENERATION_RISBench = {
    "annotation_path": "./data/annotation_256x4_8tokens_v2/RISBench_26298.json",
    "data_path": "",
}

MASK_GENERATION_RRSISD = {
    "annotation_path": "./data/annotation_256x4_8tokens_v2/RRSISD_12179.json",
    "data_path": "",
}

MASK_GENERATION_COCONUT = {
    "annotation_path": "./data/annotation_256x4_8tokens_v2/COCONUT_BASE_1597688.json",
    "data_path": "",
}

MASK_GENERATION_REFSEG = {
    "annotation_path": "./data/annotation_256x4_8tokens_v2/REFSEG_530671.json",
    "data_path": "",
}

MASK_GENERATION_RSSEG = {
    "annotation_path": "./data/annotation_256x4_8tokens_v2/RSDenseSeg_1085046.json",
    "data_path": "",
}

data_dict = {
    "MASK_GENERATION_EarthReason": MASK_GENERATION_EarthReason,
    "MASK_GENERATION_LaSeRS": MASK_GENERATION_LaSeRS,
    "MASK_GENERATION_RefsegRS": MASK_GENERATION_RefsegRS,
    "MASK_GENERATION_RISBench": MASK_GENERATION_RISBench,
    "MASK_GENERATION_RRSISD": MASK_GENERATION_RRSISD,
    "MASK_GENERATION_COCONUT": MASK_GENERATION_COCONUT,
    "MASK_GENERATION_REFSEG": MASK_GENERATION_REFSEG,
    "MASK_GENERATION_RSSEG": MASK_GENERATION_RSSEG,
}




def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list

if __name__ == "__main__":
    dataset_names = ["llava_665k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)