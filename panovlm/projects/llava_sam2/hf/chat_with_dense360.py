import torch
from PIL import Image
import os
import numpy as np
import json
from pycocotools import mask as mask_util

from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, AutoProcessor

from types import MethodType
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import GenericMask
import matplotlib.colors as mplc
def draw_instance_predictions_cache(self, labels, np_masks, jittering: bool = True):
    """
    Draw instance-level prediction results on an image.

    Args:
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        jittering: if True, in color mode SEGMENTATION, randomly jitter the colors per class
            to distinguish instances from the same class

    Returns:
        output (VisImage): image object with visualizations.
    """
    boxes = None
    scores = None
    classes = None
    keypoints = None

    masks = [GenericMask(x, self.output.height, self.output.width) for x in np_masks]


    if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
        colors = (
            [self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes]
            if jittering
            else [
                tuple(mplc.to_rgb([x / 255 for x in self.metadata.thing_colors[c]]))
                for c in classes
            ]
        )

        alpha = 0.8
    else:
        colors = None
        alpha = 0.5
    alpha = 0.0

    self.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    return self.output


def visualize(image_path, cat_masks, out_path, tags):
    if tags is None:
        left_tags = [f'{i}' for i in range(len(cat_masks))]
    else:
        left_tags = tags

    unique_tags = list(set(left_tags))
    text_prompt = ','.join(unique_tags)
    metadata = MetadataCatalog.get("__unused_ape_" + text_prompt)
    metadata.thing_classes = unique_tags
    metadata.stuff_classes = unique_tags

    result_masks = cat_masks
    input_image = read_image(image_path, format="BGR")
    visualizer = Visualizer(input_image[:, :, ::-1], metadata, instance_mode=ColorMode.IMAGE)
    visualizer.draw_instance_predictions = MethodType(draw_instance_predictions_cache, visualizer)
    vis_output = visualizer.draw_instance_predictions(labels=left_tags, np_masks=result_masks)
    output_image = vis_output.get_image()
    output_image = Image.fromarray(output_image)

    output_image.save(out_path)

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


path = "./work_dirs/dense360_hf"
model = AutoModel.from_pretrained(
    path, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda", 
    attn_implementation="flash_attention_2",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
qwen_processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=path, trust_remote_code=True)

image_file = "/media/home/zhouyikang/datasets/PANO_VLM_DATA/image/val/0ae4def03aa1f83189485f5b12cb4868.jpg"
json_file = "/media/home/zhouyikang/datasets/Insta360Img_Caption_v2_cleaned/val/0ae4def03aa1f83189485f5b12cb4868.json"
image = Image.open(image_file).convert('RGB')
ori_image_size = image.size
image = image.resize((1400, 700)) # (2048, 1024) will cause CUDA OOM

with open(json_file, 'r') as f:
    mask_caption_json_data = json.load(f)
segmentation_list = []
for item in mask_caption_json_data:
    if item['ins_id'] == 46:
        segmentation_list.append(item['segmentation'])

ori_height, ori_width = segmentation_list[0]['size']
masks = decode_mask(segmentation_list, ori_height, ori_width)
print(np.sum(masks))


# question = "<image>\nGive me the segmentation mask of <p>A man wearing a white shirt, black shorts, and a black strap across his back, standing with his back turned, positioned toward the back-right direction, close to the camera.</p>."
question = "<image>\nDescribe <region> in detail."

outputs = model.predict_forward(
    question=question,
    image=image,
    mask_prompts=masks,
    tokenizer=tokenizer,
    preprocessor=qwen_processor,
    ori_image_size=ori_image_size,
    max_new_tokens=1024,
)
print(outputs['response'])

if 'grounding_masks' in outputs and len(outputs['grounding_masks']) > 0:
    grounding_masks = np.concatenate(outputs['grounding_masks'], axis=0)
    tags = ["", ]
    visualize(image_file, grounding_masks, "test_grounding.jpg", tags)


    

