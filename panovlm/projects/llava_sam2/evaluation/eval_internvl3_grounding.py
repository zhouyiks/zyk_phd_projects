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

import torch
import torchvision
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

from pycocotools import mask as mask_util



def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, Image.Image):
        image = image_file.convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


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

def extract_integers(text):
    # integers = re.findall(r'-?\d+', text)
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    result = [float(num) if '.' in num else int(num) for num in numbers]
    return result

def main():

    eval_data_json_file = "/media/home/zhouyikang/datasets/Ego360_Entity_RefSeg_val/dense360_refseg_caption_val_gpt4o.json"
    image_root = "/media/nfs/datasets/ML1_team/Insta360PanoStruct/PanoImgStruct/PanoImgStruct_V6_20231106_youtube_with_insta_owned/images"
 
    path = 'OpenGVLab/InternVL3-78B'
    device_map = split_model('OpenGVLab/InternVL3-78B')
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    eval_results_save_root = "/media/home/zhouyikang/datasets/Ego360_Entity_RefSeg_val/eval_results/internvl3/grounding"
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
        question = f"<image>\n. Please provide the bounding box coordinate of the region this sentence describes: <ref>{expression}</ref> "\
            f"(Hint: You MUST generate the response with this format: <ref>...(put the description here)</ref><box>...(put the predicted bounding box here)</box>)"
        questions = [question] * 5

        image_path = os.path.join(image_root, item['image_file'])
        image = Image.open(image_path)
        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
        num_patches_list = [pixel_values.size(0)]*5
        pixel_values = torch.cat([pixel_values]*5, dim=0)


        responses = model.batch_chat(
            tokenizer, pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config)
        
        sam_prompt_boxes = []
        for response in responses:
            PATTERN = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')
            predict_bbox = re.findall(PATTERN, response)
            try:
                predict_bbox = [float(predict_bbox[0][0]), float(predict_bbox[0][1]), float(predict_bbox[0][2]),
                                float(predict_bbox[0][3])]
            except:
                predict_bbox = extract_integers(response)[-4:]
                if len(predict_bbox) !=4:
                    predict_bbox = [0., 0., 0., 0.]

            predict_bbox = np.array(predict_bbox)
            if predict_bbox.sum() >= 4:
                predict_bbox = predict_bbox / 1000

            ori_width, ori_height = image.size
            predict_bbox[0::2] *= ori_width
            predict_bbox[1::2] *= ori_height
            sam_prompt_boxes.append(predict_bbox.copy().astype(np.int64).tolist())

        print(sam_prompt_boxes)

        save_item = copy.deepcopy(item)
        save_item.update({
            'sam_prompt_boxes': sam_prompt_boxes,
        })

        with open(save_path, 'w') as f:
            json.dump(save_item, f)
    

if __name__ == "__main__":
    main()

