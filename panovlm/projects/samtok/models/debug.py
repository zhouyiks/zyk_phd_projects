from projects.samtok.models.sam2 import SAM2Config, VQ_SAM2Config, VQ_SAM2

import colorsys
import random
import matplotlib as mpl

def generate_distinct_bright_colors(count, saturation=0.7, value=0.9):
    colors = []
    
    hue_step = 1.0 / count
    
    for i in range(count):
        hue = i * hue_step
        
        hue += random.uniform(-hue_step * 0.3, hue_step * 0.3)
        hue %= 1.0 
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        colors.append((r, g, b))
    
    return colors


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
    
    alpha = 0.9
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


import os
import tqdm
import torch
import torchvision
from PIL import Image
import json
from pycocotools import mask as mask_utils
import numpy as np
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

import copy
from xtuner.model.utils import guess_load_checkpoint

# build vq-sam2 model
CODEBOOK_SIZE = 4096
CODEBOOK_DEPTH = 1
sam2_config = SAM2Config(
    ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
)

vq_sam2_config = VQ_SAM2Config(
    sam2_config=sam2_config,
    codebook_size=CODEBOOK_SIZE,
    codebook_depth=CODEBOOK_DEPTH,
    shared_codebook=False,
    latent_dim=256,
)
vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()
state = torch.load("/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/pretrained_weights_rs/iter_57740.pth", map_location="cpu")
state = state["state_dict"]
vq_sam2.load_state_dict(state)
sam2_image_processor = DirectResize(1024)


json_file = "/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/formatted_datasets_pano_test_split/NWPU.json"
with open(json_file, 'r') as f:
    json_data = json.load(f)
for img_idx, image_item in enumerate(json_data):
    print(f"======>>>{img_idx+1}/{len(json_data)}")
    image_file = os.path.join("/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/formatted_datasets_pano_test_split", image_item['image_file'])
    annotations = image_item['annotation']
    for anno_item in annotations:
        segms = [anno_item['segmentation'][0]]
        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size
        binary_masks = decode_mask(segms, ori_height, ori_width)

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

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

        output_image_pred = visualize(image, pred_masks, ['']*len(pred_masks))
        output_image_gt = visualize(image, binary_masks, ['']*len(binary_masks))

        output_image_gt.save(f'./visualize_4tokens/{img_idx}_gt.jpg')
        output_image_pred.save(f'./visualize_4tokens/{img_idx}_pr.jpg')




# import copy
# from xtuner.model.utils import guess_load_checkpoint

# pretrained_pth = "pretrained_weights_rs/iter_57740.pth"
# pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
# print(pretrained_state_dict.keys())
# exit(0)

# pretrained_state_dict_new = {}
# for key in pretrained_state_dict.keys():
#     new_key = copy.deepcopy(key)
#     if key.startswith('hf_model.'):
#         new_key = new_key[len('hf_model.'):]

#     pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
# torch.save(pretrained_state_dict_new, "pretrained_weights_rs/4tokens_samtok.pth")


# dir1 = "/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/formatted_datasets_pano"
# dir2 = "/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/formatted_datasets"
# dir3 = "/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/godx7/vector_plus_samtok"
# processed_datasets = []
# all_rows = []
# for json_file in os.listdir(dir1):
#     if not json_file.endswith('.json'):
#         continue
#     if json_file not in processed_datasets:
#         processed_datasets.append(json_file)
#     with open(os.path.join(dir1, json_file), 'r') as f:
#         json_data = json.load(f)
#     for item in tqdm.tqdm(json_data):
#         image_file = item['image_file']
#         annotations = item['annotation']
#         for anno_item in annotations:
#             segmentations = anno_item['segmentation']
#             for segm in segmentations:
#                 one_row = {'image_file': os.path.join(dir1, image_file), 'segmentation': segm}
#                 all_rows.append(one_row)

# for json_file in os.listdir(dir2):
#     if not json_file.endswith('.json'):
#         continue
#     if json_file not in processed_datasets:
#         processed_datasets.append(json_file)
#     else:
#         continue
#     with open(os.path.join(dir2, json_file), 'r') as f:
#         json_data = json.load(f)
#     for item in tqdm.tqdm(json_data):
#         image_file = item['image_file']
#         annotations = item['annotation']
#         for anno_item in annotations:
#             segmentations = anno_item['segmentation']
#             for segm in segmentations:
#                 one_row = {'image_file': os.path.join(dir2, image_file), 'segmentation': segm}
#                 all_rows.append(one_row)

# for json_file in os.listdir(dir3):
#     if not json_file.endswith('_samtok.json'):
#         continue
#     if json_file not in processed_datasets:
#         processed_datasets.append(json_file)
#     else:
#         continue
#     with open(os.path.join(dir3, json_file), 'r') as f:
#         json_data = json.load(f)
#     for item in tqdm.tqdm(json_data):
#         one_row = {'image_file': os.path.join(dir3, item['image_file']), 'segmentation': item['segmentation']}
#         all_rows.append(one_row)

# with open('./data/rs_mask_info_20260116.json', 'w') as f:
#     json.dump(all_rows, f, indent=4)



# # build vq-sam2 model
# CODEBOOK_SIZE = 4096
# CODEBOOK_DEPTH = 1
# sam2_config = SAM2Config(
#     ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
# )

# vq_sam2_config = VQ_SAM2Config(
#     sam2_config=sam2_config,
#     codebook_size=CODEBOOK_SIZE,
#     codebook_depth=CODEBOOK_DEPTH,
#     shared_codebook=False,
#     latent_dim=256,
# )

# vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

# sam2_image_processor = DirectResize(1024)

# image_root = "/mnt/bn/xiangtai-training-data-video/zhouyikang/MaskTokenizer/data/object365/"
# pano_image_root = "/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/coconut/xdeng77/coconut_large/panoptic_object365/"
# anno_file = "/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/coconut/xdeng77/coconut_large/panseg_object365_train_v2.json"
# with open(anno_file, 'r') as f:
#     anno_data = json.load(f)

# categories = anno_data['categories']
# anno_info_list = anno_data['annotations']

# coco_id_to_name = {meta['id']: meta['name'] for meta in categories}
# category_isthing = {meta['name']: meta['isthing'] for meta in categories}

# for anno_info in anno_info_list:
#     segments_info = anno_info['segments_info']
#     if 'object365_file_name' in anno_info:
#         image_id = anno_info['object365_file_name']
#     elif 'file_name' in anno_info:
#         image_id = anno_info['file_name']
#     else:
#         ValueError(f"image_id not found in anno_info: {anno_info}")

#     if '.png' in image_id:
#         image_id = image_id.split('.png')[0]
#     if '.jpg' in image_id:
#         image_id = image_id.split('.jpg')[0]


#     # if os.path.exists(os.path.join(temp_save_root, f"{image_id}.json")):
#     #     continue

#     patch_list = ['patch17', 'patch23', 'patch25', 'patch28', 'patch32', 'patch35', 'patch38', 'patch40', 'patch42', 'patch44', 'patch50']
#     image_path = None
#     for patch_name in patch_list:
#         if os.path.exists(os.path.join(image_root, patch_name, f"{image_id}.jpg")):
#             image_path = os.path.join(image_root, patch_name, f"{image_id}.jpg")
#             break
#     if image_path is None:
#         continue
    
#     image = Image.open(image_path).convert('RGB')
#     ori_width, ori_height = image.size
    
#     pano_file = os.path.join(pano_image_root, f"{image_id}.png")
#     mask_image = Image.open(pano_file)
#     mask_image_np = np.array(mask_image)[:, :, 0]

#     categories_name_to_masks = {}
#     for segment_info in segments_info:
#         category_id = segment_info['category_id']
#         if category_id == 0:
#             continue
#         isthing = segment_info['isthing']
#         mask = mask_image_np == segment_info['id']
#         if coco_id_to_name[category_id] not in categories_name_to_masks:
#             categories_name_to_masks[coco_id_to_name[category_id]] = []
#         categories_name_to_masks[coco_id_to_name[category_id]].append(mask)

#     turn_idx = 0
#     conversation = []
#     answer = "```json\n[{mask_2d}]\n```"
#     mask_2d_str = ''
#     class_names = []
#     for category_name, category_masks in categories_name_to_masks.items():
#         sam2_image = np.array(image)
#         sam2_image = sam2_image_processor.apply_image(sam2_image)
#         sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
#         sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

#         masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in category_masks])
#         valid_masks = masks.sum(-1).sum(-1) > 0
#         masks = masks[valid_masks]

#         if len(masks) == 0:
#             print("len(masks) == 0!!!")
#             continue

#         order = np.arange(masks.shape[0])
        
#         masks = masks[torch.as_tensor(order, dtype=torch.long)]
    
#         try:
#             boxes = torchvision.ops.masks_to_boxes(masks)
#         except Exception as e:
#             continue
        
#         whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
#         boxes = boxes / whwh
#         boxes = boxes.to(vq_sam2.device)
#         masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
#         num_ins = len(masks)
#         if num_ins < 2:
#             continue

#         end_idx = 2
#         start_idx = 0
#         vq_sam2_output = vq_sam2(
#             sam2_pixel_values.repeat(end_idx-start_idx, 1, 1, 1),
#             masks[start_idx:end_idx],
#             boxes[start_idx:end_idx],
#             reconstruct_mask=True,
#         )
        
#         exit(0)

