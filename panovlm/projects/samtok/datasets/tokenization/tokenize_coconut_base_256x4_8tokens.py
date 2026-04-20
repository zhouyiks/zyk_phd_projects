import os
import argparse
import json
import tqdm
from pycocotools import mask as mask_utils
from PIL import Image
import sys
import io

import numpy as np
import torch
import torchvision
import copy
import cv2
import pyarrow.parquet as pq


from projects.samtok.datasets.tokenization.coconut_meta import COCO_META

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

    sample_files = [
        "/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/coconut/xdeng77/coconut_b/data/train-00000-of-00004.parquet",
        "/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/coconut/xdeng77/coconut_b/data/train-00001-of-00004.parquet",
        "/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/coconut/xdeng77/coconut_b/data/train-00002-of-00004.parquet",
        "/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/coconut/xdeng77/coconut_b/data/train-00003-of-00004.parquet",
    ]

    dataset_name = "COCONUT_BASE"
    temp_save_root = f"./temp_data_256x4_8tokens/{dataset_name}"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)

    temp_save_root = f"./temp_data_256x4_8tokens/{dataset_name}_DENSESEG_TOKENIZATION_RAW_DATA"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)

    coco_id_to_name = {meta['id']: meta['name'] for meta in COCO_META}
    category_isthing = {meta['name']: meta['isthing'] for meta in COCO_META}

    all_cls_name = [meta['name'] for meta in COCO_META]
    ins_cls_name = [meta['name'] for meta in COCO_META if meta['isthing']]

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    img_count = 0
    img_shard_size = 10000
    img_shard_items = []
    img_shard_idx = 0

    # index = 0
    for sample_file in sample_files:
        parquet_file = pq.ParquetFile(sample_file)
        data = parquet_file.read().to_pandas()
        rows = data.shape[0]

        chunk_size = (rows+2) // 3
        _start_ = task_id * chunk_size
        _end_ = _start_ + chunk_size
        _end_ = rows if _end_ > rows else _end_

        subset = data.iloc[_start_:_end_]
        subset_rows = subset.shape[0]

        for row_idx in tqdm.trange(subset_rows, desc=f"Processing {os.path.basename(sample_file)}"):
            row = subset.iloc[row_idx]
            # if row_idx + 1 in [1223, 2197, 2630, 6061, 2373, 9453, 1100, 10757, 10384, 5729, 12662, 12901, 6362, 637, 13642, 7596]:
            #     continue
            masks = row['mask']
            segments_info = row['segments_info']
            image_info = row['image_info']

            image_file = image_info['file_name']
            if '.jpg' in image_file:
                image_id = image_file.split('.jpg')[0]
            elif '.png' in image_file:
                image_id = image_file.split('.png')[0]
            else:
                raise ValueError(f"Unsupported image format: {image_file}")
            
            image_path = os.path.join("/mnt/bn/xiangtai-training-data-video/zhouyikang/MaskTokenizer/data/coco/train2017", image_file)
            if not os.path.exists(image_path):
                image_path = os.path.join("/mnt/bn/xiangtai-training-data-video/zhouyikang/MaskTokenizer/data/coco/unlabeled2017", image_file)
                if not os.path.exists(image_path):
                    print(image_path, "is not found!!!")
                    continue
            image = Image.open(image_path).convert('RGB')
            ori_width, ori_height = image.size

            mask_image = Image.open(io.BytesIO(masks['bytes']))
            mask_image_np = np.array(mask_image)[:, :, 0]

            categories_name_to_masks = {}
            for segment_info in segments_info['segments_info']:
                category_id = segment_info['category_id']
                isthing = segment_info['isthing']
                mask = mask_image_np == segment_info['id']
                if coco_id_to_name[category_id] not in categories_name_to_masks:
                    categories_name_to_masks[coco_id_to_name[category_id]] = []
                categories_name_to_masks[coco_id_to_name[category_id]].append(mask)
    
            cls_name_2_mask_tokens = {}
            for category_name, category_masks in categories_name_to_masks.items():
                sam2_image = np.array(image)
                sam2_image = sam2_image_processor.apply_image(sam2_image)
                sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
                sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

                masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in category_masks])
                valid_masks = masks.sum(-1).sum(-1) > 0
                masks = masks[valid_masks]

                if len(masks) == 0:
                    print("len(masks) == 0!!!")
                    continue

                try:
                    boxes = torchvision.ops.masks_to_boxes(masks)
                except:
                    print("Error at boxes = torchvision.ops.masks_to_boxes(masks)")
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
                sam2_tokens_list = []
                for _quant_codes_ in quant_codes:
                    sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
                    sam2_tokens_list.append(sam2_tokens)

                cls_name_2_mask_tokens[category_name] = sam2_tokens_list
            
            # save raw tokenization data
            image_tokenization = {
                "all_cls_name": all_cls_name,
                "ins_cls_name": ins_cls_name,
                "image_file": image_path,
                "cls_name_2_mask_tokens": cls_name_2_mask_tokens
            }
            img_shard_items.append(image_tokenization)
            img_count += 1
            if img_count % img_shard_size == 0:
                img_shard_idx += 1
                out_path = f"./temp_data_256x4_8tokens/{dataset_name}_DENSESEG_TOKENIZATION_RAW_DATA/{dataset_name}-segment-{img_shard_idx:05d}-chunk-{task_id}.json"
                with open(out_path, "w") as f:
                    json.dump(img_shard_items, f)
                img_shard_items.clear()
                print(f"[SAVE] {out_path} ({img_count} items)", flush=True)

            # one class per conversation
            for cls_name, mask_tokens_list in cls_name_2_mask_tokens.items():
                if category_isthing[cls_name]:
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
                    out_path = f"./temp_data_256x4_8tokens/{dataset_name}/{dataset_name}-segment-{shard_idx:05d}-chunk-{task_id}.json"
                    with open(out_path, "w") as f:
                        json.dump(shard_items, f)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)
            
    if shard_items:
        shard_idx += 1
        out_path = f"./temp_data_256x4_8tokens/{dataset_name}/{dataset_name}-segment-{shard_idx:05d}-chunk-{task_id}.json"
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} ({count} items)", flush=True)

    if img_shard_items == 0:
        img_shard_idx += 1
        out_path = f"./temp_data_256x4_8tokens/{dataset_name}_DENSESEG_TOKENIZATION_RAW_DATA/{dataset_name}-segment-{img_shard_idx:05d}-chunk-{task_id}.json"
        with open(out_path, "w") as f:
            json.dump(img_shard_items, f)
        img_shard_items.clear()
        print(f"[SAVE] {out_path} ({img_count} items)", flush=True)

if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)