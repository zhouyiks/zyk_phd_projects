import pickle
import os
from typing import Literal
import json
import numpy as np
from PIL import Image
import random
import copy
import shutil
import tifffile as tf

from pycocotools import mask as mask_util

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from datasets import Dataset as HFDataset
from datasets import DatasetDict

from xtuner.dataset.huggingface import build_origin_dataset
from xtuner.registry import BUILDER
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX

from .dynamic_preprocess import dynamic_preprocess
from .constants import (CLS_TOKEN, SEG_TOKEN, PHRASE_START_TOKEN, 
    PHRASE_END_TOKEN, OBJ_START_TOKEN, OBJ_END_TOKEN, OBJ_CONTEXT_TOKEN, DEFAULT_OBJ_TOKEN)


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
    
    # alpha = 0

    self.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    return self.output


def safe_read_tiff(image_file):
    """
    安全地读取TIFF文件，处理GDAL_NODATA标签解析错误
    
    Args:
        image_file: TIFF文件路径
        
    Returns:
        numpy.ndarray: 图像数组
    """
    try:
        tiff_array = tf.imread(image_file)
    except ValueError as e:
        # 处理GDAL_NODATA标签解析错误
        if "is not castable to uint8" in str(e):
            # 使用PIL作为备选方案读取TIFF文件
            try:
                image = Image.open(image_file)
                tiff_array = np.array(image)
            except Exception as pil_error:
                # 如果PIL也失败，尝试使用tifffile但忽略错误
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tiff_array = tf.imread(image_file, ignore_tags=True)
        else:
            raise e
    
    # 处理多通道数组
    if len(tiff_array.shape) == 3 and tiff_array.shape[2] > 3:
        # 多通道图像，取前3个通道
        tiff_array = tiff_array[:, :, :3]
    elif len(tiff_array.shape) == 2:
        # 灰度图像，转换为3通道
        tiff_array = np.stack([tiff_array] * 3, axis=-1)
    
    # 确保数据类型为uint8
    if tiff_array.dtype != np.uint8:
        if tiff_array.max() <= 1.0:
            tiff_array = (tiff_array * 255).astype(np.uint8)
        else:
            tiff_array = tiff_array.astype(np.uint8)
    
    return tiff_array


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
    vis_output = visualizer.draw_instance_predictions(labels=left_tags, np_masks=result_masks)
    output_image = vis_output.get_image()
    output_image = Image.fromarray(output_image)

    return output_image

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



SEG_QUESTIONS = [
    "Can you segment all entities whose category belong to the set: {class_name}?",
    "Hey, could you run a segmentation on this image? Just focus on the entities that fall under these categories: {class_name}. Thanks!",
    "Hey, could you help me segment this image? I only need the objects that fall under these categories: {class_name}. Thanks!"
    "I need you to do a segmentation of this picture, but only for things like {class_name}.",
    "Please perform segmentation on this image. The target categories are specified in {class_name}. Ensure all relevant instances are segmented accurately.",
    "Please identify and segment all of the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with its/their segmentation mask(s)",
    "Can you segment the {class_name} in the image?",
    "Find all {class_name} in the image."
]


ANSWER_LIST = [
    "Sure, their segmentation masks can be represented by {seg_tokens}.",
    "Their corresponding segmentation masks are represented by {seg_tokens}.",
    "The segmentation masks for these classes are denoted by the tokens: {seg_tokens}.",
    "The segmentation masks are represented by {seg_tokens}.",
    "Their masks are encoded as {seg_tokens}.",
    "Here are their segmentation mask tokens: {seg_tokens}. Let me know if you need further details!",
    "Generated segmentation masks for these classes, represented by {seg_tokens}.",
    "Their segmentation masks are available as {seg_tokens}.",
    "Their segmentation masks are stored as {seg_tokens}. No segmentation was performed for classes not in the list.",
    "Segmentation mask tokens: {seg_tokens}"
]


class WHUMIXInsSegDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 tokenizer,
                 data_path,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 max_length=8192,
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 random_ratio=0.0,
                 num_m2f_queries=300,
                 dinov3_input_size=224,
                 dinov3_input_mean=(0.485, 0.456, 0.406),
                 dinov3_input_std=(0.229, 0.224, 0.225),
                 ):
        super().__init__()
        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.image_folder = image_folder
        self.template = prompt_template
        self.max_length = max_length
        self.lazy = lazy

        self.repeats = repeats
        self.random_ratio = random_ratio
        self.num_m2f_queries = num_m2f_queries
        self.dinov3_input_size = dinov3_input_size

        self._system = ''

        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int(
            (self.image_size // patch_size)**2 * (self.downsample_ratio**2))

        if preprocessor is None:
            self.transformer = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
            self.preprocessor = None
        else:
            self.transformer = None
            self.preprocessor = BUILDER.build(preprocessor)
        
        self.dinov3_min_dynamic_patch = 1
        self.dinov3_max_dynamic_patch = 17
        self.dinov3_transformer = T.Compose([
            T.ToTensor(),
            T.Resize((dinov3_input_size, dinov3_input_size), antialias=True),
            T.Normalize(
                mean=dinov3_input_mean,
                std=dinov3_input_std,
            ),
        ])

        json_data = self.json_file_preprocess(data_path)
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self._max_refetch = 1000
        self.single_image_mode = single_image_mode
    
    def json_file_preprocess(self, data_path):
        from pycocotools.coco import COCO

        coco_api = COCO(data_path)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        self.id_2_class_name = {c["id"]: c["name"] for c in sorted(cats, key=lambda x: x["id"])}

        img_ids = sorted(coco_api.imgs.keys())
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))

        dataset_dicts = []

        ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]

        num_instances_without_valid_segmentation = 0

        for img_dict, anno_dict_list in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(self.image_folder, img_dict["file_name"].split('.')[0]+'.jpg')
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.

                # The original COCO valminusminival2014 & minival2014 annotation files
                # actually contains bugs that, together with certain ways of using COCO API,
                # can trigger this assertion.
                assert anno["image_id"] == image_id

                assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

                obj = {key: anno[key] for key in ann_keys if key in anno}
                if "bbox" in obj and len(obj["bbox"]) == 0:
                    raise ValueError(
                        f"One annotation of image {image_id} contains empty 'bbox' value! "
                        "This json does not have valid COCO format."
                    )

                segm = anno.get("segmentation", None)
                if segm:  # either list[list[float]] or dict(RLE)
                    if isinstance(segm, dict):
                        if isinstance(segm["counts"], list):
                            # convert to compressed RLE
                            segm = mask_util.frPyObjects(segm, *segm["size"])
                    else:
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj["segmentation"] = segm
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts
    
    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            if self.lazy:
                cur_len = 100
            else:
                cur_len = len(data_dict['input_ids'])
                if data_dict.get('image', None) is None:
                    cur_len = -cur_len
            length_list.append(cur_len)
        return length_list * self.repeats
    
    def __len__(self):
        return len(self.text_data) * self.repeats

    def real_len(self):
        return len(self.text_data)
    
    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
    
    def prepare_data(self, index):
        data_dict = self.text_data[index]

        segmentation_list, class_names = [], []
        for ann in data_dict['annotations']:
            if self.id_2_class_name[ann['category_id']] in ['background', 'unlabeled', 'unclassified', 'Background']:
                continue
            segmentation_list.append(ann['segmentation'])
            class_names.append(self.id_2_class_name[ann['category_id']])

        if len(class_names) == 0:
            return None
        
        ori_height, ori_width = data_dict['height'], data_dict['width']
        masks = decode_mask(segmentation_list, ori_height, ori_width)

        out_data_dict = {}

        # process image
        image_file = data_dict['file_name']
        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size

        # output_image = visualize(image, masks, class_names)
        # image_name = os.path.basename(image_file).split('.')[0]
        # output_image.save(f'{image_name}.jpg')
        # exit(0)

        images, _ = dynamic_preprocess(image, self.min_dynamic_patch,
                                    self.max_dynamic_patch,
                                    self.image_size, self.use_thumbnail)
        
        pixel_values = [self.transformer(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        out_data_dict['pixel_values'] = pixel_values
        out_data_dict['image_flags'] = torch.tensor([1]* pixel_values.shape[0], dtype=torch.long)

        num_image_tokens = pixel_values.shape[0] * self.patch_token
        image_token_str = f'{self.IMG_START_TOKEN}' \
            f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
            f'{self.IMG_END_TOKEN}'

        unique_class_names = list(set(class_names))
        selected_candidate_class_names = [class_name for class_id, class_name in self.id_2_class_name.items()]
        selected_class_names = class_names
        selected_masks = masks


        if len(selected_masks) == 0:
            return None
        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in selected_masks])
        masks = masks.to(torch.uint8)
        if torch.sum(masks) == 0:
            return None
        out_data_dict['class_names'] = selected_class_names
        out_data_dict['class_name_to_contiguous_id'] = {class_name: id for id, class_name in enumerate(selected_candidate_class_names)}

        class_name_str = ""
        for class_name in selected_candidate_class_names:
            class_name_str += f"{PHRASE_START_TOKEN}{class_name}{PHRASE_END_TOKEN} {CLS_TOKEN}, "
        class_name_str = class_name_str[:-2]

        conversation = []
        question = random.choice(SEG_QUESTIONS).format(class_name=class_name_str)
        question = f'{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_OBJ_TOKEN}\n' + question
        conversation.append({'from': 'human', 'value': question})

        true_class_names_str = ""
        for cand_class_name in selected_candidate_class_names:
            if cand_class_name in selected_class_names:
                true_class_names_str += f"{PHRASE_START_TOKEN}{cand_class_name}{PHRASE_END_TOKEN} {CLS_TOKEN}, "
        true_class_names_str = true_class_names_str[:-2]
        class_ids = [out_data_dict['class_name_to_contiguous_id'][class_name] for class_name in selected_class_names]
        out_data_dict['class_ids'] = torch.as_tensor(class_ids, dtype=torch.long)

        # PROPOSAL_TOKENS = [SEG_TOKEN.format(id=str(i).zfill(3)) for i in range(self.num_m2f_queries)]
        # seg_tokens_str = ", ".join(PROPOSAL_TOKENS)
        seg_tokens_str = '[SEG]'

        response = random.choice(ANSWER_LIST).format(true_class_name=true_class_names_str, seg_tokens=seg_tokens_str)
        conversation.append({'from': 'gpt', 'value': response})

        object_token_str = f"{OBJ_START_TOKEN}"\
            f"{OBJ_CONTEXT_TOKEN * self.num_m2f_queries}"\
            f"{OBJ_END_TOKEN}"
        
        token_dict = self.get_inputid_labels(conversation, image_token_str, object_token_str)
        out_data_dict.update(token_dict)

        # dinov3 inputs
        dinov3_image = Image.open(image_file).convert("RGB")
        images, aspect_ratio = dynamic_preprocess(dinov3_image, self.dinov3_min_dynamic_patch,
                                    self.dinov3_max_dynamic_patch,
                                    self.dinov3_input_size, False)
        dinov3_pixel_values = [self.dinov3_transformer(image) for image in images]
        dinov3_pixel_values = torch.stack(dinov3_pixel_values)
        out_data_dict['dinov3_inputs'] = dinov3_pixel_values
        out_data_dict['aspect_ratio'] = aspect_ratio

        # resized_masks = torch.nn.functional.interpolate(masks.unsqueeze(0), size=(self.dinov3_input_size, self.dinov3_input_size), mode='nearest').squeeze(0)
        # out_data_dict['masks'] = resized_masks
        out_data_dict['masks'] = masks

        return out_data_dict
   
    def get_inputid_labels(self, conversations, image_token_str, object_token_str) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                if image_token_str is None and DEFAULT_IMAGE_TOKEN in msg['value']:
                    msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '')
                if DEFAULT_IMAGE_TOKEN in msg['value']:
                    msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, image_token_str).replace(DEFAULT_OBJ_TOKEN, object_token_str).strip()
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input,
                    'output': msg['value'].strip()
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_text = self.template.SEP + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        # print('len_ids: ', len(input_ids))
        return {'input_ids': input_ids, 'labels': labels}

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            real_index = index % self.real_len()
            data = self.prepare_data(real_index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data