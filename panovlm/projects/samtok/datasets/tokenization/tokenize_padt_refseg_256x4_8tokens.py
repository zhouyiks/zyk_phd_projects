import os
import argparse
import json
import tqdm
from pycocotools import mask as mask_utils
from PIL import Image
import random

import numpy as np
import torch
import torchvision
import copy

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

SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is {SEG}.",
    "Sure, {SEG}.",
    "Sure, it is {SEG}.",
    "Sure, the segmentation result is {SEG}.",
    "{SEG}.",
]

def main():
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

    temp_save_root = "./temp_data_256x4_8tokens/PADT_REFCOCO"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "PADT_REFCOCO"

    train_json_files = [
        '/mnt/bn/xiangtai-training-data-video/zhouyikang/discrete_spatial_tokenizer/data/PaDT-MLLM/RefCOCO/refcoco_train.json',
        '/mnt/bn/xiangtai-training-data-video/zhouyikang/discrete_spatial_tokenizer/data/PaDT-MLLM/RefCOCO/refcoco+_train.json',
        '/mnt/bn/xiangtai-training-data-video/zhouyikang/discrete_spatial_tokenizer/data/PaDT-MLLM/RefCOCO/refcocog_train.json',
    ]

    all_data_dict = []
    for json_file in train_json_files:
        with open(json_file, 'r') as f:
            for line in f:
                # Skip empty lines
                if line.strip():
                    all_data_dict.append(json.loads(line))

    print("===========>TOTAL_ITEMS: ", len(all_data_dict))

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    for data_dict in tqdm.tqdm(all_data_dict):
        image_name = data_dict['image']
        image_path = os.path.join('/mnt/bn/xiangtai-training-data-video/zhouyikang/MaskTokenizer/data/coco/train2014', image_name)

        image_file = os.path.basename(image_path)
        if '.jpg' in image_file:
            image_id = image_file.split('.jpg')[0]
        elif '.png' in image_file:
            image_id = image_file.split('.png')[0]
        else:
            raise ValueError(f"Unsupported image format: {image_file}")

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        instances = data_dict['objects']
        for instance in instances:
            rle = instance['rle']
            text = instance['label']

            binary_masks = decode_mask([rle], ori_height, ori_width)

            # output_image = visualize(image, binary_masks[0][np.newaxis, :, :], [""])
            # output_image.save('./padt_refcoco.jpg')

            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
            boxes = torchvision.ops.masks_to_boxes(masks)
            whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
            boxes = boxes / whwh
            boxes = boxes.to(vq_sam2.device)
            masks = [masks.to(vq_sam2.device)]
            
            with torch.no_grad():
                vq_sam2_output = vq_sam2(
                    sam2_pixel_values.repeat(len(masks), 1, 1, 1),
                    masks,
                    None,
                    reconstruct_mask=False,
                )
                quant_codes = vq_sam2_output.quant_codes
            quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = []
            for _quant_codes in quant_codes:
                _quant_codes = _quant_codes[0]
                remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
            quant_codes = remap_quant_codes

            question = random.choice(SEG_QUESTIONS).format(class_name=text)
            question = "<image>\n" + question

            sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes[0]]) + MT_END_TOKEN
            # answer = random.choice(ANSWER_LIST).format(SEG=sam2_tokens)
            answer = "```json\n[{mask_2d}]\n```"
            item_str = "{\"mask_2d\": " + sam2_tokens + ", \"label\": \"" + text + "\"}"
            answer = answer.format(mask_2d=item_str)

            conversation = []
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': answer})

            ret_data_dict = {
                'image': image_path,
                'conversations': conversation,
            }

            shard_items.append(ret_data_dict)
            count += 1

            # print(ret_data_dict)
            # exit(0)

            if count % shard_size == 0:
                shard_idx += 1
                out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}.json")
                with open(out_path, "w") as f:
                    json.dump(shard_items, f)
                shard_items.clear()
                print(f"[SAVE] {out_path} ({count} items)", flush=True)

    # 收尾
    if shard_items:
        shard_idx += 1
        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}.json")
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} (final, total={count})", flush=True) 

if __name__ == '__main__':
    main()