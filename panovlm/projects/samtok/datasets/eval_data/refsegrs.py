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

def extract_mt_token_ids(text):
    pattern = r"<mt_(\d{4})>"
    return [int(x) for x in re.findall(pattern, text)]

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
    pattern_too_many = r'(<mt_start>)(<mt_\d+>)(<mt_\d+>)(?:<mt_\d+>)+<mt_end>'
    replacement_too_many = r'\1\2\3<mt_end>'
    text = re.sub(pattern_too_many, replacement_too_many, text)
    # 规则 2: 处理标记太少的情况 (只有1个，且有<|mt_end|>)
    pattern_too_few_with_end = r'(<mt_start>)(<mt_\d+>)(<mt_end>)'
    replacement_too_few = r'\1\2<mt_9999><mt_end>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)
    # 规则 3: 处理标记太少的情况 (只有1个，且没有<|mt_end|>)
    # 使用负向前瞻确保后面不是另一个mt_token
    pattern_too_few_no_end = r'(<mt_start>)(<mt_\d+>)(?!<mt_)'
    replacement_too_few_no_end = r'\1\2<mt_9999><mt_end>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(mask_utils.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = mask_utils.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask

def _decode_prediction_masks(prediction_rles):
    decoded_masks = []
    for prediction_rle in prediction_rles:
        if prediction_rle is None:
            continue
        decoded_masks.extend(rle_to_mask([prediction_rle]))
    return decoded_masks

def _compute_best_mask_metrics(pred_masks, gt_mask, device, intersection_and_union_fn):
    gt_mask_tensor = torch.from_numpy(gt_mask).int().to(device).contiguous()
    preds_to_eval = pred_masks or [np.zeros_like(gt_mask, dtype=np.uint8)]

    max_foreground_iou = -1.0
    best_iou = None
    best_intersection = None
    best_union = None

    for pred_mask in preds_to_eval:
        pred_mask_tensor = torch.from_numpy(pred_mask).int().to(device).contiguous()
        intersection, union, _ = intersection_and_union_fn(
            pred_mask_tensor, gt_mask_tensor, 2, ignore_index=255
        )
        intersection = intersection.cpu().numpy()
        union = union.cpu().numpy()

        acc_iou = intersection / (union + 1e-5)
        acc_iou[union == 0] = 1.0
        foreground_iou = acc_iou[1]

        if foreground_iou > max_foreground_iou:
            max_foreground_iou = foreground_iou
            best_iou = acc_iou
            best_intersection = intersection
            best_union = union

    return best_intersection, best_union, best_iou

def metric():
    from projects.samtok.datasets.eval_data.utils import Summary, AverageMeter, intersectionAndUnionGPU
    
    metric_dir = './temp_save/refsegrs'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trackers = {
        "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
        "union": AverageMeter("Union", ":6.3f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
    }
    for json_file in sorted(os.listdir(metric_dir)):
        if not json_file.endswith('.json'):
            continue

        with open(os.path.join(metric_dir, json_file), 'r') as f:
            data_dict = json.load(f)

        pred_masks = _decode_prediction_masks(data_dict.get('prediction_masks', []))
        gt_masks = rle_to_mask(data_dict['gt_masks'])

        for gt_mask in gt_masks:
            intersection, union, accuracy_iou = _compute_best_mask_metrics(
                pred_masks, gt_mask, device, intersectionAndUnionGPU
            )
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=1)

    ciou = trackers["intersection"].sum[1] / (trackers["union"].sum[1] + 1e-10)
    giou = trackers["gIoU"].avg[1]

    print("ciou: {:.4f}, giou: {:.4f}".format(ciou, giou))
    return {'Acc': ciou, 'CIoU': ciou, 'GIoU': giou}


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

def main():
    args = parse_args()

    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 4
    NUM_MASK_TOKENS=8
    sam2_config = SAM2Config(
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
        num_mask_tokens=NUM_MASK_TOKENS,
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

    state = torch.load(args.vq_sam2_path, map_location="cpu")
    vq_sam2.load_state_dict(state)
    sam2_image_processor = DirectResize(1024)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    txt_path = "./data/RefSegRS/RefSegRS/output_phrase_val.txt"
    with open(txt_path, "r", encoding="utf-8") as f:
        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, text = line.split(" ", 1)
            data.append((int(idx), text))

    case_ids = list(range(len(data)))
    for case_id in tqdm.tqdm(case_ids):
        item = data[case_id]
        image_name, expression = item
        image_path = os.path.join('./data/RefSegRS/RefSegRS/images_png', f"{image_name}.png")
        mask_path = os.path.join('./data/RefSegRS/RefSegRS/masks', f'{image_name}.tif')

        label_mask = cv2.imread(mask_path, 2)
        ref_mask = np.array(label_mask) > 50
        ref_mask = ref_mask.astype(np.uint8)
        gt_mask = ref_mask[np.newaxis, :, :]
        mask_h, mask_w = gt_mask.shape[-2:]
        gt_mask = mask_to_rle(gt_mask)

        question = f"Provide a segmentation mask according to the description: {expression}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": question},
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
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        quant_ids = extract_mt_token_ids(output_text[0])
        if len(quant_ids) == 0:
            zero_mask = np.zeros((1, mask_h, mask_w)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            prediction = {'image_id': case_id, 'gt_masks': gt_mask, 'prediction_masks': zero_mask}
            with open(f"./temp_save/refsegrs/{case_id}.json", 'w') as f:
                json.dump(prediction, f)
            continue
        else:
            if len(quant_ids) % CODEBOOK_DEPTH != 0:
                print("FORMAT ERROR: ", output_text)
                output_text = [fix_mt_format_comprehensive(output_text[0])]
                print("FIXED OUTPUT TEXT: ", output_text)
                quant_ids = extract_mt_token_ids(output_text[0])
            assert len(quant_ids) % CODEBOOK_DEPTH == 0
            batch_size = len(quant_ids) // CODEBOOK_DEPTH
            remap_quant_ids = []
            for bs_id in range(batch_size):
                chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
                remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
                remap_chunk_quant_ids_error_handle = [quant_id if quant_id < CODEBOOK_SIZE else -1 for quant_id in remap_chunk_quant_ids]
                remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

            image = Image.open(image_path).convert('RGB')
            ori_width, ori_height = image.size
            sam2_image = np.array(image)
            sam2_image = sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
            sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

            quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

            with torch.no_grad():
                pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
            pred_masks = torch.nn.functional.interpolate(pred_masks, size=(mask_h, mask_w), mode='bilinear')
            pred_masks = pred_masks > 0.5
            pred_masks = pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

            pred_masks = mask_to_rle(pred_masks)
            prediction = {'image_id': case_id, 'gt_masks': gt_mask, 'prediction_masks': pred_masks}
            with open(f"./temp_save/refsegrs/{case_id}.json", 'w') as f:
                json.dump(prediction, f)

if __name__ == '__main__':
    # main()
    print(metric())
