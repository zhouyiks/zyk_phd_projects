import os
import argparse
import json
import tqdm
from pycocotools import mask as mask_utils
from PIL import Image

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

def main():
    MT_START_TOKEN = '<mt_start>'
    MT_END_TOKEN = '<mt_end>'
    MT_CONTEXT_TOKEN = '<mt_{}>'

    CODEBOOK_SIZE = 4096
    CODEBOOK_DEPTH = 1
    sam2_config = SAM2Config(
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
        num_mask_tokens=16,
        is_causal=True,
    )

    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )
    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    state = torch.load("./checkpoints_extract/rs_vq_sam2_4096x1_16tokens_true_causal_attn_v2.pth", map_location="cpu")
    vq_sam2.load_state_dict(state)

    sam2_image_processor = DirectResize(1024)

    temp_save_root = "./temp_data_16token_true_causal_attn/EarthReason"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "EarthReaon"

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    image_files = list(os.listdir('./data/EarthReason/train/images'))
    for image_file in tqdm.tqdm(image_files):
        image_path = os.path.join('./data/EarthReason/train/images', image_file)
        image_name = image_file.split('.')[0]

        mask_file = f'{image_name}.png'
        mask_path = os.path.join('./data/EarthReason/train/labels', mask_file)

        qa_file = f'{image_name}.json'
        qa_path = os.path.join('./data/EarthReason/train/QAs', qa_file)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 1
        mask = mask.astype(np.uint8)

        with open(qa_path, 'r') as f:
            qa_data = json.load(f)
        questions = qa_data['questions']
        answers = qa_data['answer']

        if np.sum(mask) == 0:
            for q in questions:
                prefix_inst = '<image>\nPlease doing Reasoning Segmentation according to the following instruction:'
                instruction = ' {}'.format(q)
                a = "There is no target object in the image."
                conversation = []
                conversation.append({'from': 'human', 'value': prefix_inst + instruction})
                conversation.append({'from': 'gpt', 'value': a})
                
                ret_data_dict = {
                    'image': image_path,
                    'conversations': conversation,
                }

                shard_items.append(ret_data_dict)
                count += 1

                if count % shard_size == 0:
                    shard_idx += 1
                    out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}.json")
                    with open(out_path, "w") as f:
                        json.dump(shard_items, f)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)
            continue

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(mask.copy()))])
        boxes = torchvision.ops.masks_to_boxes(masks)
        whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
        boxes = boxes / whwh
        boxes = boxes.to(vq_sam2.device)
        masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
        
        with torch.no_grad():
            vq_sam2_output = vq_sam2(
                sam2_pixel_values.repeat(len(masks), 1, 1, 1),
                masks,
                None,
                reconstruct_mask=False,
            )
            quant_codes = vq_sam2_output.quant_codes.cpu().numpy().astype(np.int32).tolist()[0]
            sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code[0]).zfill(4)) for code in quant_codes]) + MT_END_TOKEN

        if len(answers) == 0:
            for q in questions:
                prefix_inst = '<image>\nPlease doing Reasoning Segmentation according to the following instruction:'
                instruction = ' {}'.format(q)
                a = "There is no target object in the image."
                conversation = []
                conversation.append({'from': 'human', 'value': prefix_inst + instruction})
                conversation.append({'from': 'gpt', 'value': a})
                
                ret_data_dict = {
                    'image': image_path,
                    'conversations': conversation,
                }

                shard_items.append(ret_data_dict)
                count += 1

                if count % shard_size == 0:
                    shard_idx += 1
                    out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}.json")
                    with open(out_path, "w") as f:
                        json.dump(shard_items, f)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)
        else:
            for q in questions:
                prefix_inst = '<image>\nPlease doing Reasoning Segmentation according to the following instruction:'
                instruction = ' {}'.format(q)

                for a in answers:
                    conversation = []
                    conversation.append({'from': 'human', 'value': prefix_inst + instruction})
                    conversation.append({'from': 'gpt', 'value': f"<think>\n{a}</think>\nSure, It is {sam2_tokens}"})

                    ret_data_dict = {
                        'image': image_path,
                        'conversations': conversation,
                    }

                    shard_items.append(ret_data_dict)
                    count += 1

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