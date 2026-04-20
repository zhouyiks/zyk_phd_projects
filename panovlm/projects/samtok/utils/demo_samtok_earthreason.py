import argparse
import copy
import math
import os
import torch
import torchvision
import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import random
import re
from PIL import Image
import json
import uuid
import matplotlib as mpl
import hydra
import time
import cv2

from xtuner.model.utils import guess_load_checkpoint

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from projects.samtok.models import DirectResize
from projects.samtok.models.sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config


import colorsys

def generate_distinct_bright_colors(count, saturation=0.7, value=0.9):
    """
    生成指定数量的明显不同的亮色RGB颜色
    
    参数:
        count: 要生成的颜色数量
        saturation: 饱和度 (0-1)，值越高颜色越鲜艳
        value: 明度 (0-1)，值越高颜色越明亮
        
    返回:
        包含RGB元组的列表，每个元组包含三个0-255的整数
    """
    colors = []
    
    # 均匀分布在色相环上，确保颜色差异明显
    hue_step = 1.0 / count
    
    for i in range(count):
        # 计算色相值，均匀分布在0-1之间
        hue = i * hue_step
        
        # 随机微调色相，增加多样性但保持区分度
        hue += random.uniform(-hue_step * 0.3, hue_step * 0.3)
        hue %= 1.0  # 确保在0-1范围内
        
        # 从HSV转换到RGB (HSV颜色模型更容易控制饱和度和明度)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # 转换到0-255范围
        # r = int(r * 255)
        # g = int(g * 255)
        # b = int(b * 255)
        
        colors.append((r, g, b))
    
    return colors


from pycocotools import mask as mask_utils
from types import MethodType
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.data.detection_utils import read_image, _apply_exif_orientation, convert_PIL_to_numpy
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

    # if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
    #     colors = (
    #         [self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes]
    #         if jittering
    #         else [
    #             tuple(mplc.to_rgb([x / 255 for x in self.metadata.thing_colors[c]]))
    #             for c in classes
    #         ]
    #     )

    #     alpha = 0.8
    # else:
    #     colors = None
    #     alpha = 0.5
    
    alpha = 0.0
    colors = generate_distinct_bright_colors(len(masks))

    self.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    return self.output

def draw_polygon_cache(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=4,
        )
        self.output.ax.add_patch(polygon)
        return self.output


def draw_text_cache(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # # since the text background is dark, we don't want the text to be dark
        # color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        # color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

def _change_color_brightness_cache(self, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        # assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        # color = mplc.to_rgb(color)
        # polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        # modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        # modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        # modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        # modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        # return tuple(np.clip(modified_color, 0.0, 1.0))
        return color


def visualize(input_image, cat_masks, tags):
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
    input_image = _apply_exif_orientation(input_image)
    input_image = convert_PIL_to_numpy(input_image, "BGR")
    visualizer = Visualizer(input_image[:, :, ::-1], metadata, instance_mode=ColorMode.IMAGE)
    visualizer.draw_instance_predictions = MethodType(draw_instance_predictions_cache, visualizer)
    visualizer.draw_polygon = MethodType(draw_polygon_cache, visualizer)
    visualizer.draw_text = MethodType(draw_text_cache, visualizer)
    visualizer._change_color_brightness = MethodType(_change_color_brightness_cache, visualizer)
    vis_output = visualizer.draw_instance_predictions(labels=left_tags, np_masks=result_masks)
    output_image = vis_output.get_image()
    output_image = Image.fromarray(output_image)

    return output_image

def extract_mt_token_ids_v1(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

def extract_mt_token_ids_v2(text):
    pattern = re.compile(r'<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>')
    matches = pattern.findall(text)
    ret_list = []
    for num1, num2 in matches:
        ret_list.append(int(num1))
        ret_list.append(int(num2))
    return ret_list

def find_first_index(arr, value):
    """
    在NumPy数组中找到第一个指定值的第一个出现的索引
    
    参数:
        arr: NumPy数组
        value: 要查找的值
        
    返回:
        第一个匹配值的索引，如果没有找到则返回-1
    """
    # 使用where找到所有匹配值的索引
    indices = np.where(arr == value)[0]
    
    # 返回第一个索引，如果没有找到则返回-1
    return indices[0] if len(indices) > 0 else -1

def fix_mt_format_comprehensive(text):
    """
    全面修正 <|mt_...> 格式的函数。
    它会处理以下几种情况：
    1. 标记太少 (1个): <|mt_start|><|mt_0198|><|mt_end|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    2. 标记太少 (1个, 无end): <|mt_start|><|mt_0198|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    3. 标记太多 (3个或以上): <|mt_start|><|mt_0186|><|mt_0410|><|mt_0186|><|mt_end|> -> <|mt_start|><|mt_0186|><|mt_0410|><|mt_end|>
    4. 正确格式: <|mt_start|><|mt_0044|><|mt_0442|><|mt_end|> -> 不变
    """
    # 规则 1: 处理标记太多的情况 (3个或以上)
    # 捕获前两个，匹配掉多余的，然后用前两个重构
    pattern_too_many = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>'
    replacement_too_many = r'\1\2\3<|mt_end|>'
    text = re.sub(pattern_too_many, replacement_too_many, text)
    # 规则 2: 处理标记太少的情况 (只有1个，且有<|mt_end|>)
    pattern_too_few_with_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    replacement_too_few = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)
    # 规则 3: 处理标记太少的情况 (只有1个，且没有<|mt_end|>)
    # 使用负向前瞻确保后面不是另一个mt_token
    pattern_too_few_no_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    replacement_too_few_no_end = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="pretrained_weights/iter_175473.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--dataset',
        default='./data/PaDT-MLLM/RefCOCO/refcoco_val.json',
        help='Specify a ref dataset')
    parser.add_argument('--task_id', '--task-id', type=int, default=0)
    args = parser.parse_args()
    return args

def extract_mt_id_lists(text: str):
    block_pattern = re.compile(r"<\|mt_start\|>(?:<\|mt_\d{4}\|>)+<\|mt_end\|>")
    id_pattern = re.compile(r"<\|mt_(\d{4})\|>")

    blocks = block_pattern.findall(text)
    return [[int(x) for x in id_pattern.findall(block)] for block in blocks]

def main():
    # args = parse_args()

    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    NUM_MASK_TOKENS=1
    sam2_config = SAM2Config(
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
        num_mask_tokens=NUM_MASK_TOKENS,
        is_causal=False,
    )

    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )
    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_state_dict = guess_load_checkpoint("./checkpoints/samtok_rs_ft.pth")

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)
    sam2_image_processor = DirectResize(1024)

    # state = torch.load("./checkpoints/samtok_rs_ft.pth", map_location="cpu")
    # vq_sam2.load_state_dict(state)
    # sam2_image_processor = DirectResize(1024)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "/mnt/bn/xiangtai-training-data-video/zhouyikang/MaskTokenizer/all_checkpoints/Qwen3-VL-4B-SAMTok-co", torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained("/mnt/bn/xiangtai-training-data-video/zhouyikang/MaskTokenizer/all_checkpoints/Qwen3-VL-4B-SAMTok-co")

    image_files = list(os.listdir('./data/EarthReason/val/images'))
    for image_file in tqdm.tqdm(image_files):
        image_path = os.path.join('./data/EarthReason/val/images', image_file)
        image_name = image_file.split('.')[0]

        mask_file = f'{image_name}.png'
        mask_path = os.path.join('./data/EarthReason/val/labels', mask_file)

        qa_file = f'{image_name}.json'
        qa_path = os.path.join('./data/EarthReason/val/QAs', qa_file)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 1
        mask = mask.astype(np.uint8)
        if np.sum(mask) == 0:
            continue

        with open(qa_path, 'r') as f:
            qa_data = json.load(f)
        questions = qa_data['questions']
        answers = qa_data['answer']

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": questions[0]},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=False,  # 关闭采样，使用贪婪解码
            top_p=1.0,  # 配合do_sample=False使用
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)

        mask_tokens = extract_mt_id_lists(output_text[0])
        # print("==========>>>>mask_tokens: ", mask_tokens)
        # exit(0)
        result_mask_tokens = []
        for _mask_tokens_ in mask_tokens:
            if len(_mask_tokens_) == CODEBOOK_DEPTH:
                result_mask_tokens.append(_mask_tokens_)
        mask_tokens = result_mask_tokens
        remap_mask_tokens = []
        for _mask_tokens_ in mask_tokens:
            _remap_mask_tokens_ = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(_mask_tokens_)]
            remap_mask_tokens.append(_remap_mask_tokens_)
        mask_tokens = remap_mask_tokens

        if len(mask_tokens) == 0:
            continue

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        batch_size = len(mask_tokens)
        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

        quant_ids = torch.LongTensor(mask_tokens).to(vq_sam2.device)

        with torch.no_grad():
            pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
        pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
        pred_masks = pred_masks > 0.5
        pred_masks = pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

        output_image = visualize(image, pred_masks, ['']*len(pred_masks))
        output_image.save(os.path.join('./eval_dataset/earthReason/', image_file.replace('.', '_pred.')))

        gt_mask = mask[np.newaxis, :, :]
        gt_image = visualize(image, gt_mask, ['']*len(gt_mask))
        gt_image.save(os.path.join('./eval_dataset/earthReason/', image_file.replace('.', '_gt.')))
        
        
if __name__ == '__main__':
    main()




