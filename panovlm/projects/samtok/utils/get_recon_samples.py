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
from xtuner.model.utils import guess_load_checkpoint
import matplotlib as mpl

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

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union


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


from types import MethodType
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.data.detection_utils import read_image, _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.utils.visualizer import GenericMask
import matplotlib.colors as mplc
def draw_instance_predictions_cache(self, labels, np_masks, jittering: bool = True, color=None):
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
    
    alpha = 0.5
    # colors = generate_distinct_bright_colors(len(masks)+10)
    # print(colors)
    # exit(0)
    # colors = [(0.9, 0.34650523924536275, 0.2700000000000001),] # (0.2700000000000001, 0.9, 0.8017706045997072)
    # colors = [(0.2700000000000001, 0.9, 0.8017706045997072),]

    self.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=color,
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


def visualize(input_image, cat_masks, tags, color):
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
    vis_output = visualizer.draw_instance_predictions(labels=left_tags, np_masks=result_masks, color=color)
    output_image = vis_output.get_image()
    output_image = Image.fromarray(output_image)

    return output_image


def save_masked_png(image, pred_masks, output_path):
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert('RGB'))
    else:
        image_np = np.asarray(image)

    mask = np.asarray(pred_masks)
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=0)
    else:
        mask = mask > 0
    mask = mask.astype(np.uint8)

    rgba_image = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
    rgba_image[..., :3] = image_np
    rgba_image[..., 3] = mask * 255

    Image.fromarray(rgba_image, mode='RGBA').save(output_path)


def main():

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

    pretrained_state_dict = guess_load_checkpoint("./checkpoints/vq_sam2_256x8_8tokens_true_causal_attn.pth")

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)
    sam2_image_processor = DirectResize(1024)


    # with open("./data/lasers_mask_info_20260227.json", 'r') as f:
    #     json_data = json.load(f)
    
    # save_path = "./recon_validation/lasers"
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path, exist_ok=True)
    # os.makedirs(os.path.join(save_path, 'visualize'), exist_ok=True)

    # all_iou = []
    # count = 0
    # for item in tqdm.tqdm(json_data):
    #     image_file = item['image_file']
    #     segm = item['segmentation']

    #     image = Image.open(image_file).convert('RGB')
    #     ori_width, ori_height = image.size

    #     sam2_image = np.array(image)
    #     sam2_image = sam2_image_processor.apply_image(sam2_image)
    #     sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
    #     sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

    #     binary_masks = decode_mask([segm], ori_height, ori_width)

    #     output_image_gt = visualize(image, binary_masks, ['']*len(binary_masks), color=[(0.9, 0.34650523924536275, 0.2700000000000001),])

    #     masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
    #     try:
    #         boxes = torchvision.ops.masks_to_boxes(masks)
    #     except:
    #         continue
            
    #     whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
    #     boxes = boxes / whwh
    #     boxes = boxes.to(vq_sam2.device)
    #     masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
        
    #     with torch.no_grad():
    #         vq_sam2_output = vq_sam2(
    #             sam2_pixel_values,
    #             masks,
    #             None,
    #             reconstruct_mask=True,
    #         )
        
    #     pred_masks = vq_sam2_output.pred_masks
    #     pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
    #     pred_masks = pred_masks > 0.5
    #     pred_masks = pred_masks[0].cpu().numpy().astype(np.uint8)
    #     target_mask = masks[0].cpu().numpy().astype(np.uint8)
    #     iou = mask_iou(torch.from_numpy(target_mask), torch.from_numpy(pred_masks))
    #     all_iou.append(iou.item())

    #     output_image_pred = visualize(image, pred_masks, ['']*len(binary_masks), color=[(0.2700000000000001, 0.9, 0.8017706045997072),])

    #     int_iou = int(100*iou)
    #     image_stem = str(count).zfill(5) + '_' + os.path.splitext(os.path.basename(image_file))[0]
    #     save_masked_png(image, pred_masks, os.path.join(save_path, 'visualize', f"{image_stem}_matting_pred_iou{int_iou}.png"))
    #     save_masked_png(image, target_mask, os.path.join(save_path, 'visualize', f"{image_stem}_matting_tgt_iou{int_iou}.png"))

    #     output_image_gt.save(os.path.join(save_path, 'visualize', f"{image_stem}_overlay_tgt_iou{int_iou}.png"))
    #     output_image_pred.save(os.path.join(save_path, 'visualize', f"{image_stem}_overlay_pred_iou{int_iou}.png"))

    #     image.save(os.path.join(save_path, 'visualize', f"{image_stem}.jpg"))
    #     count += 1

    # with open(os.path.join(save_path, 'iou.json'), 'w') as f:
    #     json.dump({'avg_iou': sum(all_iou) / len(all_iou), 'iou_list': all_iou}, f, indent=4)
    # exit(0)

    with open('./formatted_datasets_test_split/Postdam.json', 'r') as f:
        json_data = json.load(f)

    save_path = "./recon_validation/Postdam_256X4_8TOKEN"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'visualize'), exist_ok=True)

    all_iou = []
    count = 0
    for item in tqdm.tqdm(json_data):
        image_file = item['image_file']
        annotation = item['annotation']

        image = Image.open(os.path.join('./formatted_datasets_test_split', image_file)).convert('RGB')
        # image = Image.open(os.path.join('./formatted_datasets_pano_test_split', image_file)).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        for anno in annotation:
            for segm in anno['segmentation']:
                binary_masks = decode_mask([segm], ori_height, ori_width)

                output_image_gt = visualize(image, binary_masks, ['']*len(binary_masks), color=[(0.9, 0.34650523924536275, 0.2700000000000001),])

                masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
                boxes = torchvision.ops.masks_to_boxes(masks)
                whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
                boxes = boxes / whwh
                boxes = boxes.to(vq_sam2.device)
                masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
                
                with torch.no_grad():
                    vq_sam2_output = vq_sam2(
                        sam2_pixel_values,
                        masks,
                        None,
                        reconstruct_mask=True,
                    )
                
                pred_masks = vq_sam2_output.pred_masks
                pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
                pred_masks = pred_masks > 0.5
                pred_masks = pred_masks[0].cpu().numpy().astype(np.uint8)
                target_mask = masks[0].cpu().numpy().astype(np.uint8)
                iou = mask_iou(torch.from_numpy(target_mask), torch.from_numpy(pred_masks))
                all_iou.append(iou.item())

                output_image_pred = visualize(image, pred_masks, ['']*len(binary_masks), color=[(0.2700000000000001, 0.9, 0.8017706045997072),])

                int_iou = int(100*iou)
                image_stem = str(count).zfill(5) + '_' + os.path.splitext(os.path.basename(image_file))[0]
                save_masked_png(image, pred_masks, os.path.join(save_path, 'visualize', f"{image_stem}_matting_pred_iou{int_iou}.png"))
                save_masked_png(image, target_mask, os.path.join(save_path, 'visualize', f"{image_stem}_matting_tgt_iou{int_iou}.png"))

                output_image_gt.save(os.path.join(save_path, 'visualize', f"{image_stem}_overlay_tgt_iou{int_iou}.png"))
                output_image_pred.save(os.path.join(save_path, 'visualize', f"{image_stem}_overlay_pred_iou{int_iou}.png"))

                image.save(os.path.join(save_path, 'visualize', f"{image_stem}.jpg"))
                count += 1

    with open(os.path.join(save_path, 'iou.json'), 'w') as f:
        json.dump({'avg_iou': sum(all_iou) / len(all_iou), 'iou_list': all_iou}, f, indent=4)

if __name__ == '__main__':
    main()