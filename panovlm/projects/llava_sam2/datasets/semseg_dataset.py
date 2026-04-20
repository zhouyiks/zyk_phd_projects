import os
from typing import Literal
import json
import numpy as np
from PIL import Image
import random
import copy
import shutil
import cv2
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


# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag


def _apply_exif_orientation(image):
    if not hasattr(image, "getexif"):
        return image
    try:
        exif = image.getexif()
    except Exception:
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
    if format is not None:
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "L":
        image = np.expand_dims(image, -1)
    elif format == "BGR":
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)
    return image


def read_image(file_name, format=None):
    with open(file_name, "rb") as f:
        image = Image.open(f)
        image = _apply_exif_orientation(image)
        return convert_PIL_to_numpy(image, format)
    raise ValueError(f"Failed to read image at: {file_name}")

class SemSegDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
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
        self.semseg_gt_folder = semseg_gt_folder
        self.template = prompt_template
        self.max_length = max_length
        self.lazy = lazy

        self.repeats = repeats
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

        json_data = self._load_semseg_data()
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self._max_refetch = 1000
        self.single_image_mode = single_image_mode

        class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        class_names = ['other', 'all buildings', 'pervious surface', 'impervious surface', 'bare soil',
                       'water', 'coniferous', 'deciduous', 'brushwood', 'vineyard', 'herbaceous vegetation', 'all agricultural lands', 'all plowed lands']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0

    def _load_semseg_data(self):
        data_list = []
        for image_file in os.listdir(self.image_folder):
            if not image_file.endswith('.png') and not image_file.endswith('.jpg'):
                continue
            image_path = os.path.join(self.image_folder, image_file)
            semseg_path = os.path.join(self.semseg_gt_folder, image_file)
            data_list.append({
                'image_file': image_path,
                'anno_file': semseg_path
            })
        return data_list
    
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

        semantic_gt = read_image(data_dict.pop("anno_file"), "L")

        class_names = []
        masks = []

        unique_class_ids_in_image = np.unique(semantic_gt)

        for cat_id in unique_class_ids_in_image:
            if isinstance(self.background_id, list):
                if cat_id in self.background_id:
                    continue
            else:
                if cat_id == self.background_id:
                    continue
            sem_mask = (semantic_gt == cat_id).astype(np.uint8)

            if np.sum(sem_mask) == 0:
                continue

            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sem_mask, connectivity=8)

            # for label_id in range(1, num_labels):
            #     component_mask = (labels == label_id).astype(np.uint8)
            #     area = np.sum(component_mask)

            #     if area < 5:
            #         continue

            #     masks.append(component_mask)
            #     class_names.append(self.id_2_class_name[cat_id])
            if sem_mask.ndim == 3 and sem_mask.shape[2] == 1:
                masks.append(sem_mask[:, :, 0])
                class_names.append(self.id_2_class_name[cat_id])
            elif sem_mask.ndim == 3 and sem_mask.shape[0] == 1:
                masks.append(sem_mask[0])
                class_names.append(self.id_2_class_name[cat_id])
            elif sem_mask.ndim == 2:
                masks.append(sem_mask)
                class_names.append(self.id_2_class_name[cat_id])
            else:
                raise ValueError(f"Invalid sem_mask shape: {sem_mask.shape}")
        
        out_data_dict = {}

        # process image
        image_file = data_dict['image_file']
        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size


        return {'image_file': image_file, 'class_names': class_names, 'masks': masks}


        # cat_masks = np.stack(masks, axis=0)
        # output_image = visualize(image, cat_masks, class_names)
        # image_name = os.path.basename(image_file)
        # output_image.save(f'stest_emseg_2_entity_{image_name}')
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

        unique_class_names = list(set(class_names))
        if isinstance(self.background_id, list):
            selected_candidate_class_names = [class_name for class_id, class_name in self.id_2_class_name.items() if class_id not in self.background_id]
        else:
            selected_candidate_class_names = [class_name for class_id, class_name in self.id_2_class_name.items() if class_id != self.background_id]
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
                true_class_names_str += f"{PHRASE_START_TOKEN}{cand_class_name} (semantic segmentation){PHRASE_END_TOKEN} {CLS_TOKEN}, "
        true_class_names_str = true_class_names_str[:-2]
        class_ids = [out_data_dict['class_name_to_contiguous_id'][class_name] for class_name in selected_class_names]
        out_data_dict['class_ids'] = torch.as_tensor(class_ids, dtype=torch.long)

        PROPOSAL_TOKENS = [SEG_TOKEN.format(id=str(i).zfill(3)) for i in range(self.num_m2f_proposals)]
        seg_tokens_str = ", ".join(PROPOSAL_TOKENS)

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


class FlairSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        class_names = ['other', 'buildings', 'pervious surface', 'impervious surface', 'bare soil',
                       'water', 'coniferous', 'deciduous', 'brushwood', 'vineyard', 'herbaceous vegetation', 'agricultural lands', 'plowed lands']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0


class CHN6CUGSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1]
        class_names = ['background', 'road']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0


class CITYOSMSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2]
        class_names = ['background', 'building', 'road']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0


class DGLCSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4, 5, 6]
        class_names = ['unknown', 'urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0


class DGROADSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1]
        class_names = ['background', 'road']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0


class DLRSDSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        class_names = ['airplane', 'bare soil', 'building', 'car', 'chaparral', 'court', 'dock', 'field', 'grass', 'mobile home', 'pavement', 'sand', 'sea', 'ship', 'tank', 'tree', 'water']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0

class EVLabSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        class_names = ['background', 'farmland', 'woodland', 'grassland', 'building', 'road', 'structure', 'digging pile', 'desert', 'water']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        # self.background_id = [0, 4, 6, 7, 8]
        self.background_ids = [0,]

class LRSNYSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1]
        class_names = ['background', 'road']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0

class OEMSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        class_names = ['background', 'bareland', 'rangeland', 'developed space', 'road', 'tree', 'water', 'agriculture land', 'building']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        # self.background_id = [0, 3, 8]
        self.background_id = [0]

class OTTAWASemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1]
        class_names = ['background', 'road']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0

class POTSDAMSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4, 5]
        class_names = ['background', 'impervious surface', 'building', 'low vegetation', 'tree', 'car']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0

class SUMMERSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        class_names = ['background', 'road', 'building', 'trees', 'grass', 'bare soil', 'water', 'railway', 'swimming pool']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        # self.background_id = [0, 2, 5, 8]
        self.background_id = [0]

class UAVIDSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4, 5]
        class_names = ['background', 'building', 'road', 'tree', 'low vegetation', 'vehicle']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0

class UDD5SemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4]
        class_names = ['other', 'vegetation', 'building', 'road', 'vehicle']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0

class UDD6SemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4, 5]
        class_names = ['other', 'façade', 'road', 'vegetation', 'vehicle', 'roof']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0

class whumixSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1]
        class_names = ['background', 'building']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0

class LoveDASemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        class_names = ['no-data','background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        # self.background_id = [0, 1, 2]
        self.background_id = [0, 1]

class LandCoverAISemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4]
        class_names = ['background', 'building', 'woodland', 'water', 'road']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = [0]

class AI4CSemSegDataset(SemSegDataset):
    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 ):
        super().__init__(
            tokenizer,
            prompt_template,
            special_tokens=special_tokens,
            image_folder=image_folder,
            semseg_gt_folder=semseg_gt_folder,
            max_length=max_length,
            arch_type=arch_type,
            preprocessor=preprocessor,
            single_image_mode=single_image_mode,
            lazy=lazy,
            repeats=repeats,
            num_m2f_queries=num_m2f_queries,
            num_m2f_proposals=num_m2f_proposals,
            m2f_input_size=m2f_input_size,
            m2f_processor=m2f_processor,
        )

        class_ids = [0, 1, 2, 3, 4]
        class_names = ['background', 'building', 'woodland', 'water', 'road']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0



class GlobalScaleDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
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
        self.semseg_gt_folder = semseg_gt_folder
        self.template = prompt_template
        self.max_length = max_length
        self.lazy = lazy

        self.repeats = repeats
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

        json_data = self._load_semseg_data()
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self._max_refetch = 1000
        self.single_image_mode = single_image_mode

        class_ids = [0, 255]
        class_names = ['other', 'road centerline']
        self.id_2_class_name = {id: class_name for id, class_name in zip(class_ids, class_names)}
        self.background_id = 0

    def _load_semseg_data(self):
        data_list = []
        for image_file in os.listdir(self.image_folder):
            if not image_file.endswith('_sat.png'):
                continue
            image_id = image_file.split('_sat.png')[0]
            image_path = os.path.join(self.image_folder, image_file)
            semseg_path = os.path.join(self.semseg_gt_folder, f"{image_id}_gt.png")
            data_list.append({
                'image_file': image_path,
                'anno_file': semseg_path
            })
        return data_list
    
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

        semantic_gt = read_image(data_dict.pop("anno_file"), "L")

        class_names = []
        masks = []

        unique_class_ids_in_image = np.unique(semantic_gt)

        for cat_id in unique_class_ids_in_image:
            if isinstance(self.background_id, list):
                if cat_id in self.background_id:
                    continue
            else:
                if cat_id == self.background_id:
                    continue
            sem_mask = (semantic_gt == cat_id).astype(np.uint8)

            if np.sum(sem_mask) == 0:
                continue

            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sem_mask, connectivity=8)

            # for label_id in range(1, num_labels):
            #     component_mask = (labels == label_id).astype(np.uint8)
            #     area = np.sum(component_mask)

            #     if area < 5:
            #         continue

            #     masks.append(component_mask)
            #     class_names.append(self.id_2_class_name[cat_id])
            if sem_mask.ndim == 3 and sem_mask.shape[2] == 1:
                masks.append(sem_mask[:, :, 0])
                class_names.append(self.id_2_class_name[cat_id])
            elif sem_mask.ndim == 3 and sem_mask.shape[0] == 1:
                masks.append(sem_mask[0])
                class_names.append(self.id_2_class_name[cat_id])
            elif sem_mask.ndim == 2:
                masks.append(sem_mask)
                class_names.append(self.id_2_class_name[cat_id])
            else:
                raise ValueError(f"Invalid sem_mask shape: {sem_mask.shape}")
        
        out_data_dict = {}

        # process image
        image_file = data_dict['image_file']
        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size


        # cat_masks = np.stack(masks, axis=0)
        # mask_image = np.stack([cat_masks[0]*255, cat_masks[0]*255, cat_masks[0]*255], axis=2)
        # mask_image = Image.fromarray(mask_image)
        # image_name = os.path.basename(image_file)
        # mask_image.save(f'test_global_scale_mask_{image_name}')
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

        unique_class_names = list(set(class_names))
        if isinstance(self.background_id, list):
            selected_candidate_class_names = [class_name for class_id, class_name in self.id_2_class_name.items() if class_id not in self.background_id]
        else:
            selected_candidate_class_names = [class_name for class_id, class_name in self.id_2_class_name.items() if class_id != self.background_id]
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

        PROPOSAL_TOKENS = [SEG_TOKEN.format(id=str(i).zfill(3)) for i in range(self.num_m2f_proposals)]
        seg_tokens_str = ", ".join(PROPOSAL_TOKENS)

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

class IndonesiaDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 tokenizer,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 semseg_gt_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 single_image_mode=False,
                 lazy=True,
                 repeats=1,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 m2f_input_size=1024,
                 m2f_processor=None,
                 class_name="road",
                 ):
        super().__init__()
        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        
        self.image_folder = image_folder
        self.semseg_gt_folder = semseg_gt_folder
        self.template = prompt_template
        self.max_length = max_length
        self.lazy = lazy

        self.repeats = repeats
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

        json_data = self._load_semseg_data()
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self._max_refetch = 1000
        self.single_image_mode = single_image_mode

        self.class_name = class_name

    def _load_semseg_data(self):
        data_list = []
        for image_file in os.listdir(self.image_folder):
            if not image_file.endswith('.tif'):
                continue
            image_path = os.path.join(self.image_folder, image_file)
            image_id = image_file.split('.tif')[0]
            semseg_path = os.path.join(self.semseg_gt_folder, f"{image_id}.json")
            data_list.append({
                'image_file': image_path,
                'anno_file': semseg_path
            })
        return data_list
    
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

        anno_file = data_dict['anno_file']
        with open(anno_file, 'r') as f:
            json_data = json.load(f)
        
        out_data_dict = {}

        # process image
        image_file = data_dict['image_file']
        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size

        segms = [item['segmentation'] for item in json_data["annotations"]]
        masks = decode_mask(segms, ori_height, ori_width)
        masks = np.sum(masks, axis=0, keepdims=True)
        class_names = [self.class_name for _ in masks]


        # image_id = os.path.basename(image_file).split('.')[0]
        # cat_masks = np.stack(masks, axis=0)
        # output_image = visualize(image, cat_masks, class_names)
        # output_image.save(f'{image_id}_{self.class_name}.jpg')
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

        unique_class_names = list(set(class_names))
        selected_candidate_class_names = unique_class_names
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

        PROPOSAL_TOKENS = [SEG_TOKEN.format(id=str(i).zfill(3)) for i in range(self.num_m2f_proposals)]
        seg_tokens_str = ", ".join(PROPOSAL_TOKENS)

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
        image = Image.open(image_file).convert('RGB')
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