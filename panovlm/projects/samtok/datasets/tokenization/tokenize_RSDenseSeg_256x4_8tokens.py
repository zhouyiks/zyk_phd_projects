import os
import argparse
import json
import tqdm
from pycocotools import mask as mask_utils
from PIL import Image
import sys

import numpy as np
import torch
import torchvision
import copy
import cv2

from projects.samtok.models.sam2 import SAM2Config, VQ_SAM2Config, VQ_SAM2

from torchvision.transforms.functional import resize, to_pil_image
class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))
    
def decode_mask(object_masks, ori_height, ori_width):
    binary_masks = []
    for object_mask in object_masks:
        if isinstance(object_mask, dict):
            if isinstance(object_mask["counts"], list):
                # convert to compressed RLE
                object_mask = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            m = mask_utils.decode(object_mask)
            m = m.astype(np.uint8).squeeze()
        elif object_mask:
            rles = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            rle = mask_utils.merge(rles)
            m = mask_utils.decode(rle).astype(np.uint8).squeeze()
        else:
            m = np.zeros((ori_height, ori_width), dtype=np.uint8)
        binary_masks.append(m)
    return binary_masks

def sort_masks_top_to_bottom_left_to_right(binary_masks):
    """
    Reorder masks by raster-scan appearance order:
    top-to-bottom first, then left-to-right.
    """
    if len(binary_masks) <= 1:
        return binary_masks

    order = []
    for idx, mask in enumerate(binary_masks):
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            order.append((np.inf, np.inf, idx))
            continue
        # np.nonzero returns coordinates in row-major order, so the first
        # foreground pixel matches the desired scan order.
        order.append((int(ys[0]), int(xs[0]), idx))

    sorted_indices = [idx for _, _, idx in sorted(order)]
    return [binary_masks[idx] for idx in sorted_indices]

def build_single_class_segmentation_prompts(cls_name, is_thing):
    if is_thing:
        return [
            f"Segment all instances of {cls_name} in this image. Return one mask for each individual object.",
            f"Identify every {cls_name} in the image and provide a separate mask for each entity.",
            f"Perform single-class segmentation for {cls_name}. Output all object masks that belong to this category.",
            f"Find all {cls_name} objects in the image and return the mask for each instance.",
            f"Segment: {cls_name}"
        ]

    return [
        f"Segment all pixels belonging to {cls_name} in this image. Return a single mask for the whole category.",
        f"Perform single-class segmentation for {cls_name}. Use one mask to cover every pixel of this category.",
        f"Identify the entire {cls_name} region in the image and provide one combined mask for all matching pixels.",
        f"Return a single segmentation mask that includes all pixels labeled as {cls_name}.",
        f"Segment with one mask: {cls_name}"
    ]




def main(task_id):
    MT_START_TOKEN = '<mt_start>'
    MT_END_TOKEN = '<mt_end>'
    MT_CONTEXT_TOKEN = '<mt_{}>'

    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 4
    sam2_config = SAM2Config(
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
        num_mask_tokens=8,
        is_causal=True,
    )

    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=512,
    )
    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    state = torch.load("./checkpoints_extract/vq_sam2_256x8_8tokens_true_causal_attn.pth", map_location="cpu")
    vq_sam2.load_state_dict(state)

    sam2_image_processor = DirectResize(1024)

    dataset_list = [
                    # 'LoveDA', 'EVLab_SS', 'LandCoverAI', 'Postdam', 
                    # 'Zurich_Summer', 
                    # 'DLRSD', 'DGLandCover', 'OEM', 'FLAIR', 'CASID',
                    # 'CITY-OSM', 
                    # 'WHU_MIX_512', 
                    # 'DeepGlobe_Road_CVPR2018', 'CHN6-CUG', 'LRSNY', 'Ottawa', 
                    # 'CrowdAI',
                    # 'UDD_5', 'UDD_6',
                    # 'BSB', 'FAST', 'FineGrip', 'NWPU', 'SIOR', 'SOTA'
                    'Globe230k',
                    'GEONRW',
                    'inria',
                    'NJDS',
                    'WHU_GID',
                    'LoveDA',
                    'SIOR',
                    'SOTA',
                    ]

    for dataset_name in dataset_list:
        temp_save_root = f"./temp_data_256x4_8tokens_v2_stage2/{dataset_name}"
        if not os.path.exists(temp_save_root):
            os.makedirs(temp_save_root)
    
        temp_save_root = f"./temp_data_256x4_8tokens_v2_stage2/{dataset_name}_DENSESEG_TOKENIZATION_RAW_DATA"
        if not os.path.exists(temp_save_root):
            os.makedirs(temp_save_root)

    root_list = ['/media/disk3/dataset/zipped_formatted_datasets']

    for ds_idx, dataset_name in enumerate(dataset_list):
        print(f"==========>>>{ds_idx+1} / {len(dataset_list)} DATASET.")

        dataset_path = ''
        for root_path in root_list:
            if os.path.exists(os.path.join(root_path, f"{dataset_name}.json")):
                dataset_path = root_path
                break
        
        if dataset_path == '':
            continue

        with open(os.path.join(dataset_path, f"{dataset_name}.json"), 'r') as f:
            json_data = json.load(f)
        
        all_cls_name = []
        ins_cls_name = []
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
                if len(segmentation) > 1 and cls_name not in ins_cls_name:
                    ins_cls_name.append(cls_name)

        chunk_size = (len(json_data)+1) // 2
        _start_ = task_id * chunk_size
        _end_ = _start_ + chunk_size
        _end_ = len(json_data) if _end_ > len(json_data) else _end_

        count = 0
        shard_size = 10000
        shard_items = []
        shard_idx = 0

        img_count = 0
        img_shard_size = 10000
        img_shard_items = []
        img_shard_idx = 0

        for img_anno in tqdm.tqdm(json_data[_start_:_end_]):
            if 'image_file' in img_anno:
                image_file = img_anno['image_file']
            else:
                image_file = img_anno['image_path']
            image_path = os.path.join(dataset_path, image_file)
            if not os.path.exists(image_path):
                print(image_path, "not exists!!!")
                continue

            image = Image.open(image_path).convert('RGB')
            ori_width, ori_height = image.size

            if 'annotation' in img_anno:
                annotation = img_anno['annotation']
            else:
                annotation = img_anno['segms']
            cls_name_2_mask_tokens = {}
            for cls_item in annotation:
                if 'class_name' in cls_item:
                    cls_name = cls_item['class_name']
                else:
                    cls_name = cls_item['class']
                segmentation = cls_item['segmentation']

                binary_masks = decode_mask(segmentation, ori_height, ori_width)
                binary_masks = sort_masks_top_to_bottom_left_to_right(binary_masks)

                sam2_image = np.array(image)
                sam2_image = sam2_image_processor.apply_image(sam2_image)
                sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
                sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

                masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
                try:
                    boxes = torchvision.ops.masks_to_boxes(masks)
                except:
                    continue
                whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
                boxes = boxes / whwh
                boxes = boxes.to(vq_sam2.device)
                masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
                num_ins = len(masks)
                
                skip_this_one = False
                try:
                    with torch.no_grad():
                        vq_sam2_output = vq_sam2(
                            sam2_pixel_values.repeat(num_ins, 1, 1, 1),
                            masks,
                            None,
                            reconstruct_mask=False,
                        )
                        quant_codes = vq_sam2_output.quant_codes
                except torch.OutOfMemoryError:
                    print("num_ins is too large: ", num_ins, "; will be split into blocks (size 10)")
                    NUM_BLOCKS = num_ins // 10
                    if NUM_BLOCKS * 10 < num_ins:
                        NUM_BLOCKS += 1
                    block_quant_codes = []
                    for block_idx in range(NUM_BLOCKS):
                        start_idx = block_idx * 10
                        end_idx = min(start_idx + 10, num_ins)
                        try:
                            with torch.no_grad():
                                vq_sam2_output = vq_sam2(
                                    sam2_pixel_values.repeat(end_idx-start_idx, 1, 1, 1),
                                    masks[start_idx:end_idx],
                                    None,
                                    reconstruct_mask=False,
                                )
                        except torch.OutOfMemoryError:
                            skip_this_one = True
                            break
                        block_quant_codes.append(vq_sam2_output.quant_codes)
                        # block_pred_masks.append(vq_sam2_output.pred_masks)
                    if skip_this_one:
                        continue
                    quant_codes = torch.cat(block_quant_codes, dim=0)
                except Exception as e:
                    continue

                quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
                remap_quant_codes = []
                for _quant_codes in quant_codes:
                    _quant_codes = _quant_codes[0]
                    remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
                quant_codes = remap_quant_codes
                assert len(quant_codes) == len(binary_masks), f"len(quant_codes): {len(quant_codes)}; len(binary_masks): {len(binary_masks)}"

                sam2_tokens_list = []
                for _quant_codes_ in quant_codes:
                    sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
                    sam2_tokens_list.append(sam2_tokens)

                cls_name_2_mask_tokens[cls_name] = sam2_tokens_list
            
            # save raw tokenization data
            image_tokenization = {
                "all_cls_name": all_cls_name,
                "ins_cls_name": ins_cls_name,
                "image_file": image_file,
                "cls_name_2_mask_tokens": cls_name_2_mask_tokens
            }
            img_shard_items.append(image_tokenization)
            img_count += 1
            if img_count % img_shard_size == 0:
                img_shard_idx += 1
                out_path = f"./temp_data_256x4_8tokens_v2_stage2/{dataset_name}_DENSESEG_TOKENIZATION_RAW_DATA/{dataset_name}-segment-{img_shard_idx:05d}-chunk-{task_id}.json"
                with open(out_path, "w") as f:
                    json.dump(img_shard_items, f)
                img_shard_items.clear()
                print(f"[SAVE] {out_path} ({img_count} items)", flush=True)

            # one class per conversation
            for cls_name, mask_tokens_list in cls_name_2_mask_tokens.items():
                if cls_name in ins_cls_name:
                    question = build_single_class_segmentation_prompts(cls_name, True)
                else:
                    question = build_single_class_segmentation_prompts(cls_name, False)

                if isinstance(question, list):
                    question = np.random.choice(question)
                
                answer = ', '.join(mask_tokens_list)
                
                conversation = []
                conversation.append({'from': 'human', 'value': "<image>\n" + question})
                conversation.append({'from': 'gpt', 'value': answer})

                ret_data_dict = {
                    'image': image_path,
                    'conversations': conversation,
                }
                shard_items.append(ret_data_dict)
                count += 1

                if count % shard_size == 0:
                    shard_idx += 1
                    out_path = f"./temp_data_256x4_8tokens_v2_stage2/{dataset_name}/{dataset_name}-segment-{shard_idx:05d}-chunk-{task_id}.json"
                    with open(out_path, "w") as f:
                        json.dump(shard_items, f)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)

        if shard_items:
            shard_idx += 1
            out_path = f"./temp_data_256x4_8tokens_v2_stage2/{dataset_name}/{dataset_name}-segment-{shard_idx:05d}-chunk-{task_id}.json"
            with open(out_path, "w") as f:
                json.dump(shard_items, f)
            shard_items.clear()
            print(f"[SAVE] {out_path} ({count} items)", flush=True)

        if img_shard_items:
            img_shard_idx += 1
            out_path = f"./temp_data_256x4_8tokens_v2_stage2/{dataset_name}_DENSESEG_TOKENIZATION_RAW_DATA/{dataset_name}-segment-{img_shard_idx:05d}-chunk-{task_id}.json"
            with open(out_path, "w") as f:
                json.dump(img_shard_items, f)
            img_shard_items.clear()
            print(f"[SAVE] {out_path} ({img_count} items)", flush=True)

if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)
