import json
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask
import numpy as np
import copy
from typing import Literal

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import torchvision.transforms as T
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from torchvision.transforms.functional import InterpolationMode
from .encode_fn import video_lisa_encode_fn
from .utils import dynamic_preprocess

from .gcg_process import glamm_openpsg_map_fn, glamm_flickr_map_fn, glamm_granf_map_fn, glamm_refcocog_map_fn

class GCGDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
    ):
        super().__init__()
        assert lazy
        self.lazy = lazy
        self.max_length = max_length

        self.arch_type = arch_type
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        elif self.arch_type == 'llava':
            self.IMG_CONTEXT_TOKEN = '<image>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''

        json_data = self.json_file_preprocess(data_path)
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self.image_folder = image_folder

        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        
        self.template_map_fn = None
        # self.template_map_fn = template_map_fn
        # if isinstance(self.template_map_fn, dict) and self.lazy:
        #     _type = self.template_map_fn['type']
        #     del self.template_map_fn['type']
        #     self.template_map_fn = _type(**self.template_map_fn)

        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        self.repeats = repeats

        self._system = ''

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
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

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

        self.arch_type = arch_type
        self.single_image_mode = single_image_mode
        self._max_refetch = 1000

        self.pharse_start_token_id = self.tokenizer.convert_tokens_to_ids('<p>')
        self.pharse_end_token_id = self.tokenizer.convert_tokens_to_ids('</p>')
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids('[CLS]')

    def json_file_preprocess(self, data_path):
        with open(data_path, 'r') as f:
            json_data = json.load(f)
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

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for seg in object_mask:
                rles = mask.frPyObjects([seg], ori_height, ori_width)
                m = mask.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

    def dataset_map_fn(self, data_dict):
        data_dict = glamm_refcocog_map_fn(data_dict)
        return data_dict

    def replace_image_str(self, data_dict, image_str):
        data_dict['conversation'][0]['input'] = \
            data_dict['conversation'][0]['input'].replace(DEFAULT_IMAGE_TOKEN, image_str)
        return data_dict

    def prepare_data(self, index):

        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])

        # parse datasets
        result = self.dataset_map_fn(data_dict)
        data_dict.update(result)

        # process image
        image_file = data_dict['image']
        image = Image.open(os.path.join(self.image_folder,
                                        image_file)).convert('RGB')
        ori_width, ori_height = image.size

        out_data_dict = {}
        out_data_dict['masks'] = self.decode_mask(data_dict['masks'], ori_height=ori_height, ori_width=ori_width)
        if hasattr(self, 'extra_image_processor'):
            g_image = np.array(image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            out_data_dict['g_pixel_values'] = g_pixel_values
        
        if self.single_image_mode:
            images = [image]
        else:
            images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        self.max_dynamic_patch,
                                        self.image_size, self.use_thumbnail)
        if self.preprocessor is not None:
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

            num_image_tokens = pixel_values.shape[0] * self.patch_token

        image_token_str = f'{self.IMG_START_TOKEN}' \
                            f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                            f'{self.IMG_END_TOKEN}'

        if self.arch_type == 'qwen':
            token_dict = self.get_inputid_labels_qwen2vl(data_dict['conversations'], image_token_str)
        else:
            token_dict = self.get_inputid_labels(data_dict['conversations'], image_token_str)
        out_data_dict.update(token_dict)

        input_ids = torch.tensor(token_dict['input_ids'])
        pharse_start_token_count = (input_ids == self.pharse_start_token_id).sum()
        pharse_end_token_count = (input_ids == self.pharse_end_token_id).sum()
        cls_token_count = (input_ids == self.cls_token_id).sum()
        
        if pharse_start_token_count == pharse_end_token_count and pharse_start_token_count == cls_token_count:
            return out_data_dict
        else:
            return None

    def get_inputid_labels_qwen2vl(self, conversations, image_token_str) -> dict:
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
                    msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, image_token_str).strip()
        
        
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
    
    def get_inputid_labels(self, conversations, image_token_str) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                if image_token_str is None and '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', '')
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', image_token_str).strip()
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
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data
class RefCOCOgGCGDataset(GCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
            arch_type=arch_type,
            preprocessor=preprocessor,
        )

    def json_file_preprocess(self, data_path):
        json_data = json.load(open(data_path))

        # convert {id: dict} to dict(..., id=xx)
        for idx in range(len(json_data)):
            id = list(json_data[idx].keys())[0]
            json_data[idx] = json_data[idx][id]
            json_data[idx].update({'id': id})
        return json_data

class GranDfGCGDataset(GCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
            arch_type=arch_type,
            preprocessor=preprocessor,
        )

    def dataset_map_fn(self, data_dict):
        data_dict = glamm_granf_map_fn(data_dict)
        return data_dict

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)

            for rle in object_mask:
                m = mask.decode(rle).astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

class OpenPsgGCGDataset(GranDfGCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
            arch_type=arch_type,
            preprocessor=preprocessor,
        )
    def dataset_map_fn(self, data_dict):
        data_dict = glamm_openpsg_map_fn(data_dict)
        return data_dict


class FlickrGCGDataset(GCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
            arch_type=arch_type,
            preprocessor=preprocessor,
        )

    def dataset_map_fn(self, data_dict):
        data_dict = glamm_flickr_map_fn(data_dict)
        return data_dict

    def json_file_preprocess(self, data_path):
        def filter_images(data_infos, min_size):
            return [i for i, info in enumerate(data_infos) if min(info['width'], info['height']) >= min_size]

        # convert {id: dict} to dict(..., id=xx)
        from pycocotools.coco import COCO
        self.coco = COCO(data_path)
        self.image_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        removed_img_count = 0
        for img_id in self.image_ids:
            info = self.coco.loadImgs([img_id])[0]
            if len(info['caption'].split(' ')) < 3:
                removed_img_count += 1
                continue
            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids), f"Non-unique annotation IDs in '{data_path}'!"
        print(f'Removed {removed_img_count} images.')
        data_infos = [data_infos[i] for i in filter_images(data_infos, min_size=32)]

        # obtain_annotations
        for data_info in data_infos:
            ann_ids = self.coco.getAnnIds(imgIds=data_info['id'])
            ann_info = self.coco.loadAnns(ann_ids)
            data_info.update({'ann_info': ann_info})
        return data_infos

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = mask.decode(object_mask).astype(np.uint8)
            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks