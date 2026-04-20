import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge all list-type JSON files under input directory into one JSON file."
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        required=True,
        help="Input directory containing JSON files.",
    )
    parser.add_argument(
        "--out",
        dest="output_name",
        required=True,
        help="Output file base name, e.g. yyy or yyy.json.",
    )
    parser.add_argument(
        "--path",
        dest="output_dir",
        required=True,
        help="Directory to save merged output JSON file.",
    )
    return parser.parse_args()


def collect_json_files(root_dir):
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(".json"):
                json_files.append(os.path.join(root, file_name))
    json_files.sort()
    return json_files


def normalize_output_stem(output_name):
    stem = output_name
    if stem.lower().endswith(".json"):
        stem = os.path.splitext(stem)[0]
    return stem


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_stem = normalize_output_stem(args.output_name)

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    merged = []
    json_files = collect_json_files(input_dir)
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON content must be a list: {file_path}")
        merged.extend(data)

    sample_count = len(merged)
    output_file_name = f"{output_stem}_{sample_count}.json"
    output_file_path = os.path.join(output_dir, output_file_name)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

    print(f"Merged {len(json_files)} files.")
    print(f"Total samples: {sample_count}")
    print(f"Saved to: {output_file_path}")


def pack():
    output_dir = "/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/data/annotation_256x4_8tokens_v2"

    input_root = "/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/temp_data_256x4_8tokens_v2"
    # input_dir = os.path.join(input_root, "COCONUT_BASE")

    # merged = []
    # json_files = collect_json_files(input_dir)
    # for file_path in json_files:
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     if not isinstance(data, list):
    #         raise ValueError(f"JSON content must be a list: {file_path}")
    #     merged.extend(data)

    # sample_count = len(merged)
    # output_file_name = f"COCONUT_BASE_{sample_count}.json"
    # output_file_path = os.path.join(output_dir, output_file_name)

    # with open(output_file_path, "w", encoding="utf-8") as f:
    #     json.dump(merged, f, ensure_ascii=False)


    # #==================================
    # merged = []
    # input_dir = os.path.join(input_root, "PADT_REFCOCO")
    # json_files = collect_json_files(input_dir)
    # for file_path in json_files:
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     if not isinstance(data, list):
    #         raise ValueError(f"JSON content must be a list: {file_path}")
    #     merged.extend(data)
    
    # input_dir = os.path.join(input_root, "REFCOCOg")
    # json_files = collect_json_files(input_dir)
    # for file_path in json_files:
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     if not isinstance(data, list):
    #         raise ValueError(f"JSON content must be a list: {file_path}")
    #     merged.extend(data)
    
    # sample_count = len(merged)
    # output_file_name = f"REFSEG_{sample_count}.json"
    # output_file_path = os.path.join(output_dir, output_file_name)

    # with open(output_file_path, "w", encoding="utf-8") as f:
    #     json.dump(merged, f, ensure_ascii=False)

    #=================================
    merged = []
    dataset_list = [
        'LoveDA', 'EVLab_SS', 'LandCoverAI', 'Postdam', 
        # 'Zurich_Summer', 
        'DLRSD', 'DGLandCover', 'OEM', 'FLAIR', 'CASID',
        'CITY-OSM', 
        'WHU_MIX_512', 
        'DeepGlobe_Road_CVPR2018', 'CHN6-CUG', 'LRSNY', 'Ottawa', 
        'CrowdAI',
        'UDD_5', 'UDD_6',
        'BSB', 'FAST', 'FineGrip',
        'NWPU', 
        'SIOR', 
        # 'SOTA'
    ]
    for ds_name in dataset_list:
        input_dir = os.path.join(input_root, ds_name)
        json_files = collect_json_files(input_dir)
        for file_path in json_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON content must be a list: {file_path}")
            merged.extend(data)
    sample_count = len(merged)
    output_file_name = f"RSDenseSeg_{sample_count}.json"
    output_file_path = os.path.join(output_dir, output_file_name)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

if __name__ == "__main__":
    # main()
    pack()
