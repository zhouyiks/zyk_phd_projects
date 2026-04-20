import json
import os
from typing import Literal
import random
import re

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
import torch.nn.functional as F


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



GCG_QUESTIONS = [
    'Could you please give me a detail description of the scene? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Can you provide a detail description of the this scene? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Please describe the contents of the scene. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Could you give a detail explanation of what can be found within this scene? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you give me a detail explanation of this scene? Please respond with interleaved segmentation masks for the corresponding phrases.',
    'Could you provide me with a detail analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]

class ERPGPT4oGCGDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

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
        sam_image_processor=None,
        use_erp_pad=True,
        prompt_template=None,
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
        
        self.template = prompt_template
        if self.arch_type == 'intern_vl':
            # self._system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
            self._system = ''
            self.template['INSTRUCTION'] = '<|user|>\n{input}<|end|><|assistant|>\n'
        elif self.arch_type == 'qwen':
            self._system = ''
        elif self.arch_type == 'llava':
            self._system = ''

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

        if sam_image_processor is not None:
            self.sam_image_processor = BUILDER.build(sam_image_processor)
        
        self.text_data = self.json_file_preprocess(data_path)

        self._max_refetch = 1000
        self.single_image_mode = single_image_mode

        self.cls_token_id = self.tokenizer.convert_tokens_to_ids('[SEG]')
    
    def json_file_preprocess(self, data_path):
        mask_caption_json_files = []
        for json_fie in os.listdir(data_path):
            mask_caption_json_files.append(os.path.join(data_path, json_fie))
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
    
    def prepare_data(self, index):
        anno_file = self.text_data[index]

        with open(anno_file, 'r') as f:
            mask_caption_dict = json.load(f)
        
        image_file = os.path.join(self.image_folder, mask_caption_dict['image_file'])
        if not os.path.exists(image_file):
            return None
        gcg_caption = mask_caption_dict['gcg_caption']
        ids_list = mask_caption_dict['ids_list']
        masks = []
        for ids in ids_list:
            if len(ids) == 0:
                continue
            grounding_segmentations = []
            for id in ids:
                for item in mask_caption_dict['segmentation_list']:
                    if item['id'] == id:
                        grounding_segmentations.append(item['segmentation'])
                        break
            if len(grounding_segmentations) == 0:
                continue
            ori_height, ori_width = grounding_segmentations[0]['size']
            grounding_masks = decode_mask(grounding_segmentations, ori_height, ori_width)
            grounding_masks = np.sum(grounding_masks, axis=0)
            masks.append(grounding_masks)
        if len(masks) == 0:
            return None
        masks = np.stack(masks, axis=0).astype(np.uint8)
        
        conversation = []
        question = random.choice(GCG_QUESTIONS).strip()
        question = f'{DEFAULT_IMAGE_TOKEN}\n' + question
        conversation.append({'from': 'human', 'value': question})

        answer = gcg_caption.replace('[CLS]', '[SEG]')
        conversation.append({'from': 'gpt', 'value': answer})

        out_data_dict = {}

        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size
        image = image.resize((1400, 700))

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
        
        # object_token_str = f"{OBJ_START_TOKEN}"\
        #     f"{OBJ_CONTEXT_TOKEN * self.num_m2f_queries}"\
        #     f"{OBJ_END_TOKEN}"
        object_token_str = ""
        
        if self.arch_type == 'qwen':
            token_dict = self.get_inputid_labels_qwen2vl(conversation, image_token_str, object_token_str)
        else:
            token_dict = self.get_inputid_labels(conversation, image_token_str, object_token_str)
        out_data_dict.update(token_dict)

        input_ids = torch.tensor(token_dict['input_ids'])
        cls_token_count = (input_ids == self.cls_token_id).sum()
        if cls_token_count != len(masks):
            return None
        
        # # mask2former inputs
        # image = Image.open(image_file)
        # if len(image.split()) != 3:
        #     image = image.convert('RGB')
        #     print(f"encounter bad image case, where its channels={len(image.split())}")
        # w, h = image.size
        # if w > h:
        #     target_size = (self.m2f_input_size, int(h/w*self.m2f_input_size))
        # else:
        #     target_size = (int(w/h*self.m2f_input_size), self.m2f_input_size)
        # resized_image = image.resize(target_size)
        # cur_w, cur_h = resized_image.size
        # padded_image = np.ones(shape=(self.m2f_input_size, self.m2f_input_size, 3), dtype=np.uint8) * 255
        # padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
        # m2f_inputs = self.m2f_processor(images=Image.fromarray(padded_image), return_tensors="pt", do_resize=False)
        # out_data_dict['m2f_inputs'] = m2f_inputs

        # resized_masks = torch.nn.functional.interpolate(masks.unsqueeze(0), size=(cur_h, cur_w), mode='nearest').squeeze(0)
        # padded_masks = torch.zeros(size=(resized_masks.shape[0], self.m2f_input_size, self.m2f_input_size), dtype=resized_masks.dtype)
        # padded_masks[:, :cur_h, :cur_w] = resized_masks
        # out_data_dict['masks'] = padded_masks

        # class_ids = torch.as_tensor(class_ids, dtype=torch.long)
        # out_data_dict['class_ids'] = class_ids

        try:
            image = Image.open(image_file).convert('RGB')
        except Exception as e:
            return None
        if hasattr(self, 'sam_image_processor'):
            g_image = np.array(image)  # for grounding
            g_image = self.sam_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            out_data_dict['g_pixel_values'] = g_pixel_values
            
            masks = torch.from_numpy(masks)
            out_data_dict['masks'] = masks

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
                if image_token_str is None and "<image>" in msg['value']:
                    msg['value'] = msg['value'].replace("<image>", '')
                if "<image>" in msg['value']:
                    msg['value'] = msg['value'].replace("<image>", image_token_str).strip()
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
