import os
import json
import cv2
import math
import numpy as np
from PIL import Image
import re
import tqdm
import sys
import copy
import base64

import torch
import torchvision
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from pycocotools import mask as mask_util


def decode_mask(object_masks, ori_height, ori_width):
    binary_masks = []
    for object_mask in object_masks:
        if isinstance(object_mask, dict):
            if isinstance(object_mask["counts"], list):
                # convert to compressed RLE
                object_mask = mask_util.frPyObjects(object_mask, ori_height, ori_width)
            m = mask_util.decode(object_mask)
            m = m.astype(np.uint8).squeeze()
        elif object_mask:
            rles = mask_util.frPyObjects(object_mask, ori_height, ori_width)
            rle = mask_util.merge(rles)
            m = mask_util.decode(rle).astype(np.uint8).squeeze()
        else:
            m = np.zeros((ori_height, ori_width), dtype=np.uint8)
        binary_masks.append(m)
    if len(binary_masks) == 0:
        binary_masks.append(np.zeros((ori_height, ori_width), dtype=np.uint8))
    masks = np.stack(binary_masks, axis=0)
    return masks


def main():

    eval_data_json_file = "/media/home/zhouyikang/datasets/Ego360_Entity_RefSeg_val/dense360_refseg_caption_val_gpt4o.json"
    image_root = "/media/nfs/datasets/ML1_team/Insta360PanoStruct/PanoImgStruct/PanoImgStruct_V6_20231106_youtube_with_insta_owned/images"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

    eval_results_save_root = "/media/home/zhouyikang/datasets/Ego360_Entity_RefSeg_val/eval_results/qwen2_5vl/captioning"
    if not os.path.exists(eval_results_save_root):
        os.makedirs(eval_results_save_root)
    
    with open(eval_data_json_file, 'r') as f:
        left_task_list = json.load(f)
    
    for item in tqdm.tqdm(left_task_list):
        case_id = item['case_id']
        save_path = os.path.join(eval_results_save_root, f"{case_id}.json")
        if os.path.exists(save_path):
            continue

        segmentation_list = [item['segmentation']]
        ori_height, ori_width = segmentation_list[0]['size']
        masks = decode_mask(segmentation_list, ori_height, ori_width)

        image_path = os.path.join(image_root, item['image_file'])
        image = Image.open(image_path)

        contours, hierarchy = cv2.findContours(masks[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.array(image).copy()
        cv2.drawContours(contour_image, contours, -1, color=(0, 255, 255), thickness=2)
        contour_image_pil = Image.fromarray(contour_image)
        resized_contour_image_pil = contour_image_pil.resize((1400, 700))
        resized_contour_image_pil.save('qwen2_5vl_captoning_input.jpg')

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": 'qwen2_5vl_captoning_input.jpg',
                    },
                    {"type": "text", "text": "This is a ERP image. Give me a detail caption of the entity inside the cyan-colored contours."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

        save_item = copy.deepcopy(item)
        save_item.update({
            'prediction': output_text
        })

        with open(save_path, 'w') as f:
            json.dump(save_item, f)
    

if __name__ == "__main__":
    main()

