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
import ast

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

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def main():

    eval_data_json_file = "/media/home/zhouyikang/datasets/Ego360_Entity_RefSeg_val/dense360_refseg_caption_val_gpt4o.json"
    image_root = "/media/nfs/datasets/ML1_team/Insta360PanoStruct/PanoImgStruct/PanoImgStruct_V6_20231106_youtube_with_insta_owned/images"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

    eval_results_save_root = "/media/home/zhouyikang/datasets/Ego360_Entity_RefSeg_val/eval_results/qwen2_5vl/grounding"
    if not os.path.exists(eval_results_save_root):
        os.makedirs(eval_results_save_root)
    
    with open(eval_data_json_file, 'r') as f:
        left_task_list = json.load(f)
    
    for item in tqdm.tqdm(left_task_list):
        case_id = item['case_id']
        save_path = os.path.join(eval_results_save_root, f"{case_id}.json")
        if os.path.exists(save_path):
            continue

        expression = item['expression']
        image_path = os.path.join(image_root, item['image_file'])
        image = Image.open(image_path)
        resized_image = image.resize((1400, 700))
        resized_image.save('qwen2_5vl_grounding_input.jpg')

        sam_prompt_boxes = []
        for _ in range(5):

            prompt = f"Locate the entity described by this sentence: {expression}.\n output its bbox coordinates using JSON format."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": 'qwen2_5vl_grounding_input.jpg',
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]


            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            input_height = inputs['image_grid_thw'][0][1]*14
            input_width = inputs['image_grid_thw'][0][2]*14

            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            bounding_boxes = parse_json(output_text[0])
            try:
                json_output = ast.literal_eval(bounding_boxes)
            except Exception as e:
                end_idx = bounding_boxes.rfind('"}') + len('"}')
                truncated_text = bounding_boxes[:end_idx] + "]"
                json_output = ast.literal_eval(truncated_text)

            save_item = copy.deepcopy(item)
            save_item.update({
                'prediction': output_text
            })

            with open(save_path, 'w') as f:
                json.dump(save_item, f)
            
            image = Image.open('qwen2_5vl_grounding_input.jpg')
            width, height = image.size
            for i, bounding_box in enumerate(json_output):
                # Convert normalized coordinates to absolute coordinates
                abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
                abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
                abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
                abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1

                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1
                sam_prompt_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
        
        print(sam_prompt_boxes)

        save_item = copy.deepcopy(item)
        save_item.update({
            'sam_prompt_boxes': sam_prompt_boxes,
        })

        with open(save_path, 'w') as f:
            json.dump(save_item, f)
    
    

if __name__ == "__main__":
    main()

