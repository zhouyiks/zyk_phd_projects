import os
import sys
import collections
import os.path as osp
import random
import copy
from typing import Dict, List
from PIL import Image
import numpy as np
import torch
import torchvision
from pycocotools import mask as mask_utils
import json
import tqdm
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import io
import uuid

from projects.llava_sam2.rs_datasets.sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config

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

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union

def sort_mask_indices(masks_t: torch.Tensor, mode: str = "ltr-ttb") -> np.ndarray:
    """
    根据实例的几何位置给出排序索引。
    Args:
        masks_t: [N, H, W] 的 torch.bool/uint8 张量（每个实例一个二值mask）
        mode:
            - "ltr-ttb": left-to-right, then top-to-bottom（先按x中心，再按y中心）
            - "ttb-ltr": top-to-bottom, then left-to-right（先按y中心，再按x中心）
            - "tlbr":    purely by top-left (y1, x1) 先y后x（行优先）
    Returns:
        order: numpy 数组，形状 [N]，是重排索引
    """
    # 利用bbox/中心点作为排序依据（更稳定、无需遍历像素）
    boxes = torchvision.ops.masks_to_boxes(masks_t)  # [N,4] (x1,y1,x2,y2)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    xc = ((x1 + x2) * 0.5).cpu().numpy()
    yc = ((y1 + y2) * 0.5).cpu().numpy()
    y1n = y1.cpu().numpy()
    x1n = x1.cpu().numpy()

    if mode == "ltr-ttb":
        # 先x后y：主键x_center，次键y_center
        order = np.lexsort((yc, xc))
    elif mode == "ttb-ltr":
        # 先y后x：主键y_center，次键x_center
        order = np.lexsort((xc, yc))
    elif mode == "tlbr":
        # 以bbox左上角先y后x（更像“逐行”）
        order = np.lexsort((x1n, y1n))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return order


QUESTION_LIST = [
    "<image>\nSegment every instance that belongs to the following categories: {class_name}",
    "<image>\nLocate every instance that belongs to the following categories: {class_name}. Report segmentation masks in JSON format."
]


def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import gc
    gc.collect()


def main(task_id):
    
    # from xtuner.model.utils import guess_load_checkpoint
    # pretrained_state_dict = guess_load_checkpoint("/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/pretrained_weights/iter_73320.pth")

    # pretrained_state_dict_new = {}
    # for key in pretrained_state_dict.keys():
    #     new_key = copy.deepcopy(key)
    #     if key.startswith('hf_model.'):
    #         new_key = new_key[len('hf_model.'):]
    #     pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    # torch.save(pretrained_state_dict_new, "pretrained_weights/rs_mask_tokenizer_1024x2_v2.pth")
    # exit(0)


    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'
    
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    temp_save_root = "./temp_data_1024x2/rs_data/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "rs_data"

    sam2_config = SAM2Config(
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )
    CODEBOOK_SIZE = 1024
    CODEBOOK_DEPTH = 2
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    state = torch.load("pretrained_weights/rs_mask_tokenizer_1024x2.pth", map_location="cpu")
    vq_sam2.load_state_dict(state)

    sam2_image_processor = DirectResize(1024)

    json_file = "/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/data/rs_ov_seg_data_image_info_v2.json"

    with open(json_file, 'r') as f:
        data_dict_list = json.load(f)
        for idx, data_dict in enumerate(data_dict_list):
            data_dict.update({'idx': idx})
        rows = len(data_dict_list)

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    chunk_size = (rows+31) // 32
    _start_ = task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    for data_dict in tqdm.tqdm(data_dict_list[_start_:_end_]):
        image_path = data_dict['image_file']
        image_name = image_path.split('.')[0]
        anno_list = data_dict['annotation']
        item_idx = data_dict['idx']

        output_path = os.path.join(temp_save_root, f"{image_name}_{item_idx}.json")
        if os.path.exists(output_path):
            continue

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        categories_name_to_masks = {}
        for anno in anno_list:
            cls_name = anno['class_name']
            rle = anno['segmentation']
            mask = decode_mask([rle], ori_height, ori_width)[0]
            if cls_name not in categories_name_to_masks:
                categories_name_to_masks[cls_name] = []
            categories_name_to_masks[cls_name].append(mask)


        conversation = []
        answer = "```json\n[{mask_2d}]\n```"
        mask_2d_str = ''
        class_names = []
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
                order = sort_mask_indices(masks, mode="ltr-ttb")
            except:
                order = np.arange(masks.shape[0])
            
            masks = masks[torch.as_tensor(order, dtype=torch.long)]

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
                        boxes,
                        reconstruct_mask=False,
                    )
                    quant_codes = vq_sam2_output.quant_codes.detach()
            except:
                print("num_ins is too large: ", num_ins, "; will be split into blocks (size 10)")
                NUM_BLOCKS = num_ins // 10
                if NUM_BLOCKS * 10 < num_ins:
                    NUM_BLOCKS += 1
                block_quant_codes = []
                for block_idx in range(NUM_BLOCKS):
                    start_idx = block_idx * 10
                    end_idx = min(start_idx + 10, num_ins)
                    with torch.no_grad():
                        try:
                            vq_sam2_output = vq_sam2(
                                sam2_pixel_values.repeat(end_idx-start_idx, 1, 1, 1),
                                masks[start_idx:end_idx],
                                boxes[start_idx:end_idx],
                                reconstruct_mask=False,
                            )
                            block_quant_codes.append(vq_sam2_output.quant_codes.detach())
                        except:
                            print("sam2_pixel_values[start_idx:end_idx].shape: ", sam2_pixel_values.repeat(end_idx-start_idx, 1, 1, 1).shape)
                            print("masks[start_idx:end_idx].shape: ", [_mask_.shape for _mask_ in masks[start_idx:end_idx]])
                            print("boxes[start_idx:end_idx].shape: ", [_box_.shape for _box_ in boxes[start_idx:end_idx]])
                            skip_this_one = True
                            break
                            # exit(0)
                if skip_this_one:
                    print("skip this one!!!")
                    continue
                quant_codes = torch.cat(block_quant_codes, dim=0)
            
            if len(quant_codes) == 0:
                continue

            quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = []
            for _quant_codes in quant_codes:
                _quant_codes = _quant_codes[0]
                remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
            quant_codes = remap_quant_codes

            if skip_this_one:
                print("skip this one")
                continue
            
            for _quant_codes_ in quant_codes:
                sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
                item_str = "{\"mask_2d\": " + sam2_tokens + ", \"label\": \"" + category_name + "\"}"
                mask_2d_str += item_str + ",\n"
                if category_name not in class_names:
                    class_names.append(category_name)
        
        if mask_2d_str == '':
            continue

        mask_2d_str = mask_2d_str[:-len(",\n")]
        answer = answer.format(mask_2d=mask_2d_str)

        category_name_str = ', '.join(class_names)
        question = random.choice(QUESTION_LIST).format(class_name=category_name_str)

        conversation.append({'from': 'human', 'value': question})
        conversation.append({'from': 'gpt', 'value': answer})
        
        ret_data_dict = {
            'image': image_path,
            'conversations': conversation,
        }

        with open(output_path, 'w') as f:
            json.dump(ret_data_dict, f)



    #     shard_items.append(ret_data_dict)
    #     count += 1

    #     if count % shard_size == 0:
    #         shard_idx += 1
    #         out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}.json")
    #         with open(out_path, "w") as f:
    #             json.dump(shard_items, f)
    #         shard_items.clear()
    #         print(f"[SAVE] {out_path} ({count} items)", flush=True)

    # # 收尾
    # if shard_items:
    #     shard_idx += 1
    #     out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}.json")
    #     with open(out_path, "w") as f:
    #         json.dump(shard_items, f)
    #     shard_items.clear()
    #     print(f"[SAVE] {out_path} (final, total={count})", flush=True) 


if __name__ == "__main__":
    # task_id = sys.argv[1]
    # task_id = int(task_id)
    # main(task_id)

    # all_data_dict = []
    # for json_file in os.listdir("./temp_data_1024x2/rs_data/"):
    #     with open(os.path.join("./temp_data_1024x2/rs_data/", json_file), 'r') as f:
    #         json_data = json.load(f)
    #         all_data_dict.append(json_data)
    # with open("./data/mask_generation_1024x2_v1.json", 'w') as f:
    #     json.dump(all_data_dict, f)

    with open("./data/mask_generation_1024x2_v1.json", "r") as f:
        json_data = json.load(f)
    print(len(json_data))
    print(json_data[0])
