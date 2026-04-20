import json
import os
from typing import Literal
import random

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask as mask_util
import numpy as np
import copy

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import torchvision.transforms as T
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from torchvision.transforms.functional import InterpolationMode
from .encode_fn import video_lisa_encode_fn
from .utils import dynamic_preprocess
from .utils.constants import (CLS_TOKEN, SEG_TOKEN, PHRASE_START_TOKEN, 
    PHRASE_END_TOKEN, OBJ_START_TOKEN, OBJ_END_TOKEN, OBJ_CONTEXT_TOKEN, DEFAULT_OBJ_TOKEN)

import torch.nn.functional as F


DETAILED_QUESTIONS =  [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give about the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]

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

class ERPRegionCaptionDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    LIMIT = ''

    VP_START_TOKEN = '<vp>'
    VP_END_TOKEN = '</vp>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    def __init__(
        self,
        image_folder,
        data_path=None,
        tokenizer=None,
        max_length=8196,
        special_tokens=None,
        arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
        preprocessor=None,
        single_image_mode=False,
        lazy=True,
        repeats=1,
        num_m2f_queries=300,
        num_m2f_proposals=100,
        m2f_input_size=1024,
        m2f_processor=None,
        single_region_mode=True,
        use_erp_pad=True,
    ):
        super().__init__()
        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.image_folder = image_folder
        self.data_path = data_path

        self.max_length = max_length
        self.lazy = lazy
        self.repeats = repeats
        
        self.num_m2f_queries = num_m2f_queries
        self.num_m2f_proposals = num_m2f_proposals
        self.m2f_input_size = m2f_input_size
        self.m2f_processor = BUILDER.build(m2f_processor)

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
            if use_erp_pad:
                self.IMG_CONTEXT_TOKEN = '<|erp_pad|>'
            else:
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

        self.text_data = self.json_file_preprocess(data_path)

        self._max_refetch = 1000
        self.single_image_mode = single_image_mode
        self.single_region_mode = single_region_mode

    
    def json_file_preprocess(self, data_path):
        mask_caption_json_files = []
        for split_name in os.listdir(data_path):
            split_path = os.path.join(data_path, split_name)
            for json_file in os.listdir(split_path):
                mask_caption_json_files.append(os.path.join(split_path, json_file))
        return mask_caption_json_files
    
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
    
    def _process_conversation(self, descriptions, region_pixels):
        n_regions = len(descriptions)
        if n_regions > 1:
            start_region_str = '<image>\n<obj_tokens>\nThere are {} part regions in the image: '.format(n_regions)
            for i in range(n_regions):
                start_region_str = start_region_str + f"region{i+1}" +\
                    self.VP_START_TOKEN + self.IMG_CONTEXT_TOKEN * region_pixels[i] + self.VP_END_TOKEN
                if i == n_regions - 1:
                    start_region_str = start_region_str + '.\n'
                else:
                    start_region_str = start_region_str + ', '
            converations = []
            for i, description in enumerate(descriptions):
                question = random.choice(DETAILED_QUESTIONS).strip().replace('<region>', f"region{i+1}") + self.LIMIT
                if i == 0:
                    question = start_region_str + question
                converations.append({'from': 'human', 'value': question})
                converations.append({'from': 'gpt', 'value': description})
            return converations
        else:
            start_region_str = '<image>\n<obj_tokens>\n'
            question = random.choice(DETAILED_QUESTIONS).strip() + self.LIMIT
            question = start_region_str + question.replace('<region>', self.VP_START_TOKEN + self.IMG_CONTEXT_TOKEN * region_pixels[0] + self.VP_END_TOKEN)
            converations = []
            converations.append({'from': 'human', 'value': question})
            converations.append({'from': 'gpt', 'value': descriptions[0]})
            return converations
    
    def _get_region_infos(self, masks, size):
        masks = F.interpolate(
            masks.unsqueeze(0),
            size=size,
            mode='nearest').squeeze(0)
        region_pixels = []
        for mask in masks:
            region_pixels.append(mask.bool().to(torch.int64).sum())
        return masks, region_pixels

    def prepare_data(self, index):
        anno_file = self.text_data[index]

        with open(anno_file, 'r') as f:
            mask_caption_dict = json.load(f)
        
        image_file = os.path.join(self.image_folder, mask_caption_dict['image_file'])
        if not os.path.exists(image_file):
            return None
        segmentation_list, captions = [], []
        for item in mask_caption_dict['annotations']:
            segmentation_list.append(item['segmentation'])
            captions.append(item['caption'])
        
        ori_height, ori_width = segmentation_list[0]['size']
        masks = decode_mask(segmentation_list, ori_height, ori_width)
        
        selected_captions, selected_masks = [], []
        NUM_REGION = 1 if self.single_region_mode else 5
        indices = list(range(len(masks)))
        random.shuffle(indices)
        for idx in indices[:NUM_REGION]:
            selected_captions.append(captions[idx])
            selected_masks.append(masks[idx])

        out_data_dict = {}

        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size
        # image = image.resize((1400, 700))
        
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
                # _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                # _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                num_image_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                merge_length = self.preprocessor.image_processor.merge_size ** 2
                out_data_dict['image_flags'] = torch.tensor(
                    [1] * (_data_dict['image_grid_thw'][0, 1] * _data_dict['image_grid_thw'][0, 2] // merge_length), dtype=torch.long)
                prompt_mask_size = (_data_dict['image_grid_thw'][0, 1]//self.preprocessor.image_processor.merge_size, _data_dict['image_grid_thw'][0, 2]//self.preprocessor.image_processor.merge_size)
            elif self.arch_type == 'llava':
                _data_dict = self.preprocessor(images, do_resize=True, size=(self.image_size, self.image_size))
                _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                num_image_tokens = _data_dict['pixel_values'].shape[0] * self.patch_token
                out_data_dict['image_flags'] = torch.tensor([1] * _data_dict['pixel_values'].shape[0])
                prompt_mask_size = (int(self.patch_token**0.5), int(self.patch_token**0.5))
            else:
                raise NotImplementedError
            out_data_dict.update(_data_dict)
        else:
            pixel_values = [self.transformer(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            out_data_dict['pixel_values'] = pixel_values
            out_data_dict['image_flags'] = torch.tensor([1]* pixel_values.shape[0], dtype=torch.long)
            num_image_tokens = pixel_values.shape[0] * self.patch_token
            prompt_mask_size = (int(self.image_size // self.patch_size * self.downsample_ratio),
                int(self.image_size // self.patch_size * self.downsample_ratio))

            vp_overall_mask = torch.Tensor([False] * (len(images) - 1) + [True])
            out_data_dict['vp_overall_mask'] = vp_overall_mask

        selected_masks = np.stack(selected_masks, axis=0)
        selected_masks_tensor = torch.from_numpy(selected_masks)
        prompt_masks_tensor, region_pixels = self._get_region_infos(selected_masks_tensor, prompt_mask_size)

        if self.arch_type == "qwen":
            vp_second_per_grid_ts = [0] * len(selected_masks)
            vp_grid_thw = [out_data_dict['image_grid_thw']]*len(selected_masks)
            out_data_dict['vp_second_per_grid_ts'] = vp_second_per_grid_ts
            out_data_dict['vp_grid_thw'] = vp_grid_thw

        # out_data_dict['prompt_masks'] = [prompt_masks_tensor_i.unsqueeze(0) for prompt_masks_tensor_i in prompt_masks_tensor]
        out_data_dict['prompt_masks'] = prompt_masks_tensor

        image_token_str = f'{self.IMG_START_TOKEN}' \
            f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
            f'{self.IMG_END_TOKEN}'
        
        object_token_str = f"{OBJ_START_TOKEN}"\
            f"{OBJ_CONTEXT_TOKEN * self.num_m2f_queries}"\
            f"{OBJ_END_TOKEN}"
        # object_token_str = ""
        
        conversation = self._process_conversation(selected_captions, region_pixels)
        if self.arch_type == 'qwen':
            token_dict = self.get_inputid_labels_qwen2vl(conversation, image_token_str, object_token_str)
        else:
            token_dict = self.get_inputid_labels(conversation, image_token_str, object_token_str)
        out_data_dict.update(token_dict)

        if self.arch_type == 'qwen':
            return out_data_dict
         
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
        out_data_dict['m2f_masks'] = padded_masks

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