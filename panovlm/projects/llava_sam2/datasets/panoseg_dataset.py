import os
from typing import Literal
import json
import numpy as np
from PIL import Image
import random
import copy
import shutil

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from datasets import Dataset as HFDataset
from datasets import DatasetDict

from xtuner.dataset.huggingface import build_origin_dataset
from xtuner.registry import BUILDER
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX

from .coco_category import COCO_CATEGORIES, FineGrip_CATEGORIES
from .utils.dynamic_preprocess import dynamic_preprocess
from .utils.constants import (CLS_TOKEN, SEG_TOKEN, PHRASE_START_TOKEN, 
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


# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag



SEG_QUESTIONS = [
    "Segment from the class prompt: {class_name}",
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

# ANSWER_LIST = [
#     "This image contains \"{true_class_name}\", their segmentation masks can be represented by {seg_tokens}.",
#     "The image contains the following classes from your list: {true_class_name}. Their corresponding segmentation masks are represented by {seg_tokens}.",
#     "In this image, I have identified the following classes from your provided list: {true_class_name}. The segmentation masks for these classes are denoted by the tokens: {seg_tokens}.",
#     "As requested, I have segmented the objects in the image belonging to the classes: {true_class_name}. The segmentation masks are represented by {seg_tokens}.",
#     "Segmentation completed. The detected classes from your list are: {true_class_name}, and their masks are encoded as {seg_tokens}.",
#     "Based on your input, I found the following classes in the image: {true_class_name}. Here are their segmentation mask tokens: {seg_tokens}. Let me know if you need further details!",
#     "Step 1: Identified classes from your list: {true_class_name}.\nStep 2: Generated segmentation masks for these classes, represented by {seg_tokens}.",
#     "The image includes objects from the classes: {true_class_name}. Their segmentation masks are available as {seg_tokens}.",
#     "From the provided class list, the following classes were found in the image: {true_class_name}. Their segmentation masks are stored as {seg_tokens}. No segmentation was performed for classes not in the list.",
#     "Here\'s what I found in the image: the classes {true_class_name} are present, and their segmentation masks are encoded as {seg_tokens}. Hope this helps!",
#     "Results:\n\tDetected classes: {true_class_name}\n\tSegmentation mask tokens: {seg_tokens}"
# ]

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


# [{'image_file': xxx, 'annotations': [{'category': xx, 'segmentation': {}, }, ...]}, ...]



def _get_coco_panoptic_meta():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES]
    thing_colors = [k["color"] for k in COCO_CATEGORIES]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        else:
            stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta

def _get_finegrip_meta():
    meta = {}

    thing_classes = [k["name"] for k in FineGrip_CATEGORIES]
    thing_colors = [k["color"] for k in FineGrip_CATEGORIES]
    stuff_classes = [k["name"] for k in FineGrip_CATEGORIES]
    stuff_colors = [k["color"] for k in FineGrip_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(FineGrip_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        else:
            stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def load_finegrip_panoptic_json(json_file, image_dir, gt_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    coco_id_to_name = {e['id']: e['name'] for e in FineGrip_CATEGORIES}

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_name"] = coco_id_to_name[segment_info["category_id"]]
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_name"] = coco_id_to_name[segment_info["category_id"]]
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info
    
    with open(json_file, 'r') as f:
        json_info = json.load(f)
    
    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, os.path.splitext(ann["file_name"])[0] + ".png")
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    return ret

def load_coco_panoptic_json(json_file, image_dir, gt_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    coco_id_to_name = {e['id']: e['name'] for e in COCO_CATEGORIES}

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_name"] = coco_id_to_name[segment_info["category_id"]]
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_name"] = coco_id_to_name[segment_info["category_id"]]
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info
    
    with open(json_file, 'r') as f:
        json_info = json.load(f)
    
    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    return ret

def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image

def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        image = _apply_exif_orientation(image)
        return convert_PIL_to_numpy(image, format)
    raise ValueError(f"Failed to read image at: {file_name}")


class CoCoPanoSegDataset(Dataset):
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
                 pano_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 random_ratio=0.0,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__()
        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.image_folder = image_folder
        self.pano_gt_folder = pano_gt_folder
        self.template = prompt_template
        self.max_length = max_length
        self.lazy = lazy

        self.repeats = repeats
        self.random_ratio = random_ratio
        self.num_m2f_queries = num_m2f_queries
        self.num_m2f_proposals = num_m2f_proposals
        self.m2f_input_size = m2f_input_size
        self.m2f_processor = BUILDER.build(m2f_processor)

        self._system = ''

        self.arch_type = arch_type
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int(
            (self.image_size // patch_size)**2 * (self.downsample_ratio**2))

        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

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

        json_data = self.json_file_preprocess(data_path)
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self._max_refetch = 1000
        self.single_image_mode = single_image_mode


    def json_file_preprocess(self, data_path):
        coco_pano_meta = _get_coco_panoptic_meta()

        json_data = load_coco_panoptic_json(data_path, self.image_folder, self.pano_gt_folder, coco_pano_meta)
        return json_data
    
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

        # decode masks
        pan_seg_gt = read_image(data_dict.pop("pan_seg_file_name"), "RGB")
        segments_info = data_dict["segments_info"]

        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)

        class_names = []
        masks = []
        for segment_info in segments_info:
            class_name = segment_info["category_name"]
            if not segment_info["iscrowd"]:
                class_names.append(class_name)
                masks.append(pan_seg_gt == segment_info["id"])

        out_data_dict = {}

        # process image
        image_file = data_dict['file_name']
        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size

        if self.single_image_mode:
            images = [image]
        else:
            images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        self.max_dynamic_patch,
                                        self.image_size, self.use_thumbnail)
        
        if self.preprocessor is not None and self.single_image_mode:
            if self.arch_type == 'qwen':
                image_processor = copy.deepcopy(self.preprocessor.image_processor)
                
                _data_dict = image_processor.preprocess(image, return_tensors="pt")
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                num_image_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                merge_length = self.preprocessor.image_processor.merge_size ** 2
                out_data_dict['image_flags'] = torch.tensor(
                    [1] * (_data_dict['image_grid_thw'][0, 1] * _data_dict['image_grid_thw'][0, 2] // merge_length), dtype=torch.long)
            elif self.arch_type == 'llava':
                _data_dict = self.preprocessor(images, do_resize=True, size=(self.image_size, self.image_size))
                _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                num_image_tokens = _data_dict['pixel_values'].shape[0] * self.patch_token
                out_data_dict['image_flags'] = torch.tensor([1] * _data_dict['pixel_values'].shape[0])
            else:
                raise NotImplementedError
            out_data_dict.update(_data_dict)
        else:
            pixel_values = [self.transformer(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            out_data_dict['pixel_values'] = pixel_values
            out_data_dict['image_flags'] = torch.tensor([1]* pixel_values.shape[0], dtype=torch.long)

            num_image_tokens = pixel_values.shape[0] * self.patch_token
        image_token_str = f'{self.IMG_START_TOKEN}' \
            f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
            f'{self.IMG_END_TOKEN}'
        
        # conversation
        unique_class_names = list(set(class_names))
        # if len(unique_class_names) > 2 and random.random() < self.random_ratio:
        #     random_num = random.randint(1, len(unique_class_names))
        #     random_select_class_names = np.random.choice(unique_class_names, random_num)
        #     # exclude_class_names = [class_name for class_name in unique_class_names if class_name not in random_select_class_names]
        #     selected_candidate_class_names = [e['name'] for e in COCO_CATEGORIES if e['name'] in random_select_class_names]
        #     selected_class_names, selected_masks = [], []
        #     for class_name, mask in zip(class_names, masks):
        #         if class_name in random_select_class_names:
        #             selected_class_names.append(class_name)
        #             selected_masks.append(mask)
        if random.random() < self.random_ratio:
            # reduce the number of candidate class names
            exclude_class_names = [e['name'] for e in COCO_CATEGORIES if not e['name'] in unique_class_names]
            random_num = random.randint(1, len(exclude_class_names))
            random_select_class_names = np.random.choice(exclude_class_names, random_num)
            selected_candidate_class_names = [e['name'] for e in COCO_CATEGORIES if e['name'] in random_select_class_names or e['name'] in unique_class_names]
            selected_class_names = class_names
            selected_masks = masks
        else:
            selected_candidate_class_names = [e['name'] for e in COCO_CATEGORIES]
            selected_class_names = class_names
            selected_masks = masks


        # add stuff or thing note
        isthing_dict = {e['name']: e['isthing'] for e in COCO_CATEGORIES}
        new_selected_class_names = []
        for class_name in selected_class_names:
            if isthing_dict[class_name] == 1:
                new_selected_class_names.append(f"{class_name} (thing)")
            else:
                new_selected_class_names.append(f"{class_name} (stuff)")
        selected_class_names = new_selected_class_names
        new_selected_candidate_class_names = []
        for class_name in selected_candidate_class_names:
            if isthing_dict[class_name] == 1:
                new_selected_candidate_class_names.append(f"{class_name} (thing)")
            else:
                new_selected_candidate_class_names.append(f"{class_name} (stuff)")
        selected_candidate_class_names = new_selected_candidate_class_names

        # out_data_dict['masks'] = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in selected_masks])
        if len(selected_masks) == 0:
            return None
        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in selected_masks])
        masks = masks.to(torch.uint8)
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
        
        # version 0: generate seg tokens
        PROPOSAL_TOKENS = [SEG_TOKEN.format(id=str(i).zfill(3)) for i in range(self.num_m2f_proposals)]
        seg_tokens_str = ", ".join(PROPOSAL_TOKENS)

        # version 1: use mask2former queries directly
        # seg_tokens_str = SEG_TOKEN.format(id="")

        response = random.choice(ANSWER_LIST).format(true_class_name=true_class_names_str, seg_tokens=seg_tokens_str)
        conversation.append({'from': 'gpt', 'value': response})

        object_token_str = f"{OBJ_START_TOKEN}"\
            f"{OBJ_CONTEXT_TOKEN * self.num_m2f_queries}"\
            f"{OBJ_END_TOKEN}"
        
        if self.arch_type == 'qwen':
            token_dict = self.get_inputid_labels_qwen2vl(conversation, image_token_str, object_token_str)
        else:
            token_dict = self.get_inputid_labels(conversation, image_token_str, object_token_str)
        out_data_dict.update(token_dict)

        # mask2former inputs
        image = Image.open(image_file)
        if len(image.split()) != 3:
            image = image.convert('RGB')
            print(f"encounter bad image case, where its channels={len(image.split())}")
        w, h = image.size
        if w > h:
            target_size = (self.m2f_input_size, int(h/w*self.m2f_input_size))
        else:
            target_size = (int(w/h*self.m2f_input_size), self.m2f_input_size)
        resized_image = image.resize(target_size)
        cur_w, cur_h = resized_image.size
        padded_image = np.ones(shape=(self.m2f_input_size, self.m2f_input_size, 3), dtype=np.uint8) * 255
        padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
        m2f_inputs = self.m2f_processor(images=Image.fromarray(padded_image), return_tensors="pt", do_resize=False)
        out_data_dict['m2f_inputs'] = m2f_inputs

        resized_masks = torch.nn.functional.interpolate(masks.unsqueeze(0), size=(cur_h, cur_w), mode='nearest').squeeze(0)
        padded_masks = torch.zeros(size=(resized_masks.shape[0], self.m2f_input_size, self.m2f_input_size), dtype=resized_masks.dtype)
        padded_masks[:, :cur_h, :cur_w] = resized_masks
        out_data_dict['masks'] = padded_masks

        # print('pixel_values.shape: ', m2f_inputs['pixel_values'].shape)
        # print("pixel_mask.shape: ", m2f_inputs['pixel_mask'].shape)
        # pixel_values.shape:  torch.Size([2, 3, 1024, 1024])
        # pixel_mask.shape:  torch.Size([2, 1024, 1024])

        return out_data_dict

    def get_inputid_labels_qwen2vl(self, conversations, image_token_str, object_token_str) -> dict:
        roles = {"human": "user", "gpt": "assistant"}
        system_message = "You are a helpful assistant."

        tokenizer = copy.deepcopy(self.tokenizer)
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                if image_token_str is None and DEFAULT_IMAGE_TOKEN in msg['value']:
                    msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '')
                if DEFAULT_IMAGE_TOKEN in msg['value']:
                    msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, image_token_str).replace(DEFAULT_OBJ_TOKEN, object_token_str).strip()
        
        
        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in conversations:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]
            
            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        if len(input_id) > self.max_length:
            input_id = input_id[:self.max_length]
            target = target[:self.max_length]
        return {'input_ids': input_id, 'labels': target}

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



class FineGripPanoSegDataset(Dataset):
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
                 pano_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 random_ratio=0.0,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__()
        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.image_folder = image_folder
        self.pano_gt_folder = pano_gt_folder
        self.template = prompt_template
        self.max_length = max_length
        self.lazy = lazy

        self.repeats = repeats
        self.random_ratio = random_ratio
        self.num_m2f_queries = num_m2f_queries
        self.num_m2f_proposals = num_m2f_proposals
        self.m2f_input_size = m2f_input_size
        self.m2f_processor = BUILDER.build(m2f_processor)

        self._system = ''

        self.arch_type = arch_type
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        if self.arch_type == 'llava':
            self.downsample_ratio = 1
        self.image_size = 448
        if self.arch_type == 'llava':
            self.image_size = 336
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int(
            (self.image_size // patch_size)**2 * (self.downsample_ratio**2))

        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

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

        json_data = self.json_file_preprocess(data_path)
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self._max_refetch = 1000
        self.single_image_mode = single_image_mode


    def json_file_preprocess(self, data_path):
        coco_pano_meta = _get_finegrip_meta()

        json_data = load_finegrip_panoptic_json(data_path, self.image_folder, self.pano_gt_folder, coco_pano_meta)
        return json_data
    
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

        # decode masks
        pan_seg_gt = read_image(data_dict.pop("pan_seg_file_name"), "RGB")
        segments_info = data_dict["segments_info"]

        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)

        class_name_2_id = {e['name']: e['id'] for e in FineGrip_CATEGORIES}

        class_names = []
        masks = []
        for segment_info in segments_info:
            class_name = segment_info["category_name"]
            if not segment_info["iscrowd"]:
                # if class_name_2_id[class_name] < 20:
                #     class_name = "Airplane (thing)"
                # else:
                #     class_name = f"{class_name} (stuff)"
                class_names.append(class_name)
                masks.append(pan_seg_gt == segment_info["id"])

        out_data_dict = {}

        # process image
        image_file = data_dict['file_name']
        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size

        # return {'image_file': image_file, 'class_names': class_names, 'masks': masks}
        
        # cat_masks = np.stack(masks, axis=0)
        # output_image = visualize(image, masks, class_names)
        # image_name = os.path.basename(image_file).split('.')[0]
        # output_image.save(f'test_read_finegrip_{image_name}.jpg')
        # exit(0)

        if self.single_image_mode:
            images = [image]
        else:
            images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        self.max_dynamic_patch,
                                        self.image_size, self.use_thumbnail)
        
        if self.preprocessor is not None and self.single_image_mode:
            if self.arch_type == 'qwen':
                image_processor = copy.deepcopy(self.preprocessor.image_processor)
                
                _data_dict = image_processor.preprocess(image, return_tensors="pt")
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                num_image_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                merge_length = self.preprocessor.image_processor.merge_size ** 2
                out_data_dict['image_flags'] = torch.tensor(
                    [1] * (_data_dict['image_grid_thw'][0, 1] * _data_dict['image_grid_thw'][0, 2] // merge_length), dtype=torch.long)
            elif self.arch_type == 'llava':
                _data_dict = self.preprocessor(images, do_resize=True, size=(self.image_size, self.image_size))
                _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                num_image_tokens = _data_dict['pixel_values'].shape[0] * self.patch_token
                out_data_dict['image_flags'] = torch.tensor([1] * _data_dict['pixel_values'].shape[0])
            else:
                raise NotImplementedError
            out_data_dict.update(_data_dict)
        else:
            pixel_values = [self.transformer(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            out_data_dict['pixel_values'] = pixel_values
            out_data_dict['image_flags'] = torch.tensor([1]* pixel_values.shape[0], dtype=torch.long)

            num_image_tokens = pixel_values.shape[0] * self.patch_token
        image_token_str = f'{self.IMG_START_TOKEN}' \
            f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
            f'{self.IMG_END_TOKEN}'
        
        # conversation
        unique_class_names = list(set(class_names))
        # if len(unique_class_names) > 2 and random.random() < self.random_ratio:
        #     random_num = random.randint(1, len(unique_class_names))
        #     random_select_class_names = np.random.choice(unique_class_names, random_num)
        #     # exclude_class_names = [class_name for class_name in unique_class_names if class_name not in random_select_class_names]
        #     selected_candidate_class_names = [e['name'] for e in COCO_CATEGORIES if e['name'] in random_select_class_names]
        #     selected_class_names, selected_masks = [], []
        #     for class_name, mask in zip(class_names, masks):
        #         if class_name in random_select_class_names:
        #             selected_class_names.append(class_name)
        #             selected_masks.append(mask)

        selected_candidate_class_names =  []
        for e in FineGrip_CATEGORIES:
            if e['id'] < 20:
                selected_candidate_class_names.append("Airplane (thing)")
            else:
                selected_candidate_class_names.append(f"{e['name']} (stuff)")
        selected_class_names = class_names
        selected_masks = masks

        # out_data_dict['masks'] = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in selected_masks])
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
        
        # version 0: generate seg tokens
        PROPOSAL_TOKENS = [SEG_TOKEN.format(id=str(i).zfill(3)) for i in range(self.num_m2f_proposals)]
        seg_tokens_str = ", ".join(PROPOSAL_TOKENS)

        # version 1: use mask2former queries directly
        # seg_tokens_str = SEG_TOKEN.format(id="")

        response = random.choice(ANSWER_LIST).format(true_class_name=true_class_names_str, seg_tokens=seg_tokens_str)
        conversation.append({'from': 'gpt', 'value': response})

        object_token_str = f"{OBJ_START_TOKEN}"\
            f"{OBJ_CONTEXT_TOKEN * self.num_m2f_queries}"\
            f"{OBJ_END_TOKEN}"
        
        if self.arch_type == 'qwen':
            token_dict = self.get_inputid_labels_qwen2vl(conversation, image_token_str, object_token_str)
        else:
            token_dict = self.get_inputid_labels(conversation, image_token_str, object_token_str)
        out_data_dict.update(token_dict)

        # mask2former inputs
        image = Image.open(image_file)
        if len(image.split()) != 3:
            image = image.convert('RGB')
            print(f"encounter bad image case, where its channels={len(image.split())}")
        w, h = image.size
        if w > h:
            target_size = (self.m2f_input_size, int(h/w*self.m2f_input_size))
        else:
            target_size = (int(w/h*self.m2f_input_size), self.m2f_input_size)
        resized_image = image.resize(target_size)
        cur_w, cur_h = resized_image.size
        padded_image = np.ones(shape=(self.m2f_input_size, self.m2f_input_size, 3), dtype=np.uint8) * 255
        padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
        m2f_inputs = self.m2f_processor(images=Image.fromarray(padded_image), return_tensors="pt", do_resize=False)
        out_data_dict['m2f_inputs'] = m2f_inputs

        resized_masks = torch.nn.functional.interpolate(masks.unsqueeze(0), size=(cur_h, cur_w), mode='nearest').squeeze(0)
        padded_masks = torch.zeros(size=(resized_masks.shape[0], self.m2f_input_size, self.m2f_input_size), dtype=resized_masks.dtype)
        padded_masks[:, :cur_h, :cur_w] = resized_masks
        out_data_dict['masks'] = padded_masks

        # print('pixel_values.shape: ', m2f_inputs['pixel_values'].shape)
        # print("pixel_mask.shape: ", m2f_inputs['pixel_mask'].shape)
        # pixel_values.shape:  torch.Size([2, 3, 1024, 1024])
        # pixel_mask.shape:  torch.Size([2, 1024, 1024])

        return out_data_dict

    def get_inputid_labels_qwen2vl(self, conversations, image_token_str, object_token_str) -> dict:
        roles = {"human": "user", "gpt": "assistant"}
        system_message = "You are a helpful assistant."

        tokenizer = copy.deepcopy(self.tokenizer)
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                if image_token_str is None and DEFAULT_IMAGE_TOKEN in msg['value']:
                    msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '')
                if DEFAULT_IMAGE_TOKEN in msg['value']:
                    msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, image_token_str).replace(DEFAULT_OBJ_TOKEN, object_token_str).strip()
        
        
        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in conversations:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]
            
            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        if len(input_id) > self.max_length:
            input_id = input_id[:self.max_length]
            target = target[:self.max_length]
        return {'input_ids': input_id, 'labels': target}

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