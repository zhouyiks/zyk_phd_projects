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

from openai import OpenAI

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

    eval_data_json_file = "./dense360_refseg_caption_val_gpt4o.json"
    image_root = "./PANO_VLM_DATA/image"

    eval_results_save_root = "./eval_results/chatgpt4o/captioning"
    if not os.path.exists(eval_results_save_root):
        os.makedirs(eval_results_save_root)
    
    with open(eval_data_json_file, 'r') as f:
        left_task_list = json.load(f)
    
    MODEL='chatgpt-4o-latest'
    client = OpenAI(api_key='')
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
        contour_image_pil.save('chatgpt4o_captoning_input.jpg')

        with open('chatgpt4o_captoning_input.jpg', "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        question = "This is a ERP image. Give me a detail caption of the entity inside the cyan-colored contours."
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": 
                [{"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}}]
                }
            ]
        )
        caption = response.choices[0].message.content
        print(caption)

        save_item = copy.deepcopy(item)
        save_item.update({
            'prediction': caption
        })

        with open(save_path, 'w') as f:
            json.dump(save_item, f)
    

if __name__ == "__main__":
    main()
