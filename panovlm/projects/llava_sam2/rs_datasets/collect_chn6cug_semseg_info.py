import os
import json
from transformers import AutoTokenizer, AutoImageProcessor
from xtuner.utils import PROMPT_TEMPLATE
from pycocotools import mask as mask_utils
import numpy as np
import tqdm 

from projects.llava_sam2.datasets.semseg_dataset import CHN6CUGSemSegDataset

def main():
    tokenizer = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path='./OpenGVLab/InternVL3-2B',
        trust_remote_code=True,
        padding_side='right')
    
    template = "internlm2_chat"
    prompt_template = PROMPT_TEMPLATE.internlm2_chat
    num_m2f_proposals = 100
    PROPOSAL_TOKENS = ['[SEG{id}]'.format(id=str(i).zfill(3)) for i in range(num_m2f_proposals)]
    special_tokens = ['<p>', '</p>', '[CLS]', '[BG_CLS]', '<obj>', '</obj>', '<OBJ_CONTEXT>'] + PROPOSAL_TOKENS
    mask2former_path = "./facebook/mask2former-swin-large-coco-panoptic"

    mask2former_processor = dict(
        type=AutoImageProcessor.from_pretrained,
        pretrained_model_name_or_path=mask2former_path,
    )

    chn6cug_semseg_dataset = dict(
        type=CHN6CUGSemSegDataset,
        image_folder="./data/koreyoshieo/rs_seg_semantic/chn6-cug/train/image",
        semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic/chn6-cug/train/mask",
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        special_tokens=special_tokens,
        arch_type='internvl',
        lazy=True,
        max_length=8192,
        repeats=1,
        num_m2f_queries=100,
        num_m2f_proposals=num_m2f_proposals,
        m2f_input_size=1024,
        m2f_processor=mask2former_processor,
    )

    all_items = []
    for idx in tqdm.tqdm(list(range(len(chn6cug_semseg_dataset)))):
        data_dict = chn6cug_semseg_dataset[idx]

        image_file = data_dict['image_file']
        class_names = data_dict['class_names']
        masks = data_dict['masks']

        category_masks = {}
        for cls_name, mask in zip(class_names, masks):
            rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            if cls_name not in category_masks:
                category_masks[cls_name] = []
            category_masks[cls_name].append(rle)
        
        for cls, rles in category_masks.items():
            ret_items = {
                'image_file': image_file,
                'class_name': cls,
                'rles': rles,
                'type': 'sem',
            }
            all_items.append(ret_items)
        
    with open('./ov_seg_data/CHN6CUGSemSegDataset.json', 'r') as f:
        json.dump(all_items, f)
    
if __name__ == "__main__":
    main()