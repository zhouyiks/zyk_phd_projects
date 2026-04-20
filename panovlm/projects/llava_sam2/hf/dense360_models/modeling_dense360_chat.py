import warnings
from typing import Any, List, Optional, Tuple, Union, Dict, Set
import copy

import transformers
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (Qwen2_5_VLPreTrainedModel, Qwen2_5_VLForConditionalGeneration)
from transformers.modeling_utils import PreTrainedModel

import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import resize, to_pil_image
import torch.nn.functional as F

from .configuration_dense360_chat import Dense360ChatConfig
from .sam2 import SAM2

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class Dense360ChatModel(PreTrainedModel):
    config_class = Dense360ChatConfig
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True

    def __init__(self, config: Dense360ChatConfig, mllm_model=None):
        super().__init__(config)

        if mllm_model is not None:
            self.mllm = mllm_model
        else:
            self.mllm = Qwen2_5_VLForConditionalGeneration._from_config(config.mllm_config)
        
        self.grounding_encoder = SAM2()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = self.mllm.model.config.hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        self.extra_image_processor = DirectResize(target_length=1024, )

        self.init_prediction_config = False

    def get_erp_rope_idx(
        self, 
        L: int,
        llm_grid_h: int,
        llm_grid_w: int,
        llm_grid_t: int,
    ):
        '''
        L: the last token idx
        '''
        gamma_1 = 1

        theta_degrees = torch.tensor([(h + 0.5) / llm_grid_h * 180 - 90 for h in range(llm_grid_h)], dtype=torch.float32)
        theta_radians = torch.deg2rad(theta_degrees)
        cos_theta = torch.cos(theta_radians)
        gamma_2 = llm_grid_h / cos_theta.sum().item()
 
        W_tensor = torch.arange(llm_grid_w)+1
        # reloc_W_tensor = (llm_grid_w-2)/4*torch.cos(torch.pi*2/llm_grid_w*W_tensor - torch.pi) + (llm_grid_w+2)/4
        # reloc_W_tensor = torch.arange(llm_grid_w) + 1 
        if llm_grid_w % 2 == 1:
            half_w = (llm_grid_w - 1) // 2
            reloc_W_tensor = torch.cat([W_tensor[0:1], W_tensor[1:1+half_w], torch.flip(W_tensor[1:1+half_w], dims=(0,))])
        else:
            half_w = (llm_grid_w - 1) // 2
            reloc_W_tensor = torch.cat([W_tensor[0:1], W_tensor[1:1+half_w+1], torch.flip(W_tensor[1:1+half_w], dims=(0,))])
        H_tensor = torch.arange(llm_grid_h) + 1
        # zero_h = H_tensor[0]
        # zero_w = reloc_W_tensor[0]
        H_tensor = H_tensor.view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        reloc_W_tensor = reloc_W_tensor.view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        cos_theta = cos_theta.view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        
        h_index = H_tensor * gamma_1
        w_index = cos_theta * reloc_W_tensor * gamma_2

        # delta_h = h_index - zero_h
        # delta_w = w_index - zero_w
        # length = delta_h ** 2 + delta_w ** 2
        # max_index = torch.argmax(length)
        # max_h_index = h_index[max_index]
        # max_w_index = w_index[max_index]

        # beta_1 = L + (llm_grid_h*llm_grid_w+1)/2 - (max_h_index+1)*gamma_1/2
        # beta_2 = L + (llm_grid_h*llm_grid_w+1)/2 - (max_w_index+1)*gamma_2/2

        beta_1 = beta_2 = L

        h_index = beta_1 + h_index
        w_index = beta_2 + w_index

        return h_index, w_index
    
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        vp_grid_thw: Optional[torch.LongTensor] = None,
        vp_second_per_grid_ts: Optional[List] = None,
        prompt_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.mllm.model.config.vision_config.spatial_merge_size
        image_token_id = self.mllm.model.config.image_token_id
        video_token_id = self.mllm.model.config.video_token_id
        vision_start_token_id = self.mllm.model.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float32,
                device=input_ids.device,
            )
            image_index, video_index, vp_index = 0, 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                erp_nums = (vision_tokens == self.erp_context_token_id).sum()
                
                vp_nums = 0
                vp_start_indices = torch.argwhere(input_ids == self.vp_start_token_id).squeeze(1)
                vision_tokens = input_ids[vp_start_indices+1]
                image_vp_nums = (vision_tokens == image_token_id).sum()
                video_vp_nums = (vision_tokens == video_token_id).sum()
                erp_vp_nums = (vision_tokens == self.erp_context_token_id).sum()
                vp_nums = image_vp_nums + video_vp_nums + erp_vp_nums

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_erps, remain_vps = image_nums, video_nums, erp_nums, vp_nums
                for _ in range(image_nums + video_nums + erp_nums + vp_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if self.erp_context_token_id in input_tokens and remain_erps > 0:
                        ed_erp = input_tokens.index(self.erp_context_token_id, st)
                    else:
                        ed_erp = len(input_tokens) + 1
                    if self.vp_start_token_id in input_tokens and remain_vps > 0:
                        ed_vp = input_tokens.index(self.vp_start_token_id, st) + 1
                        assert input_tokens
                    else:
                        ed_vp = len(input_tokens) + 1
                    
                    vp_type = 'none'
                    if ed_image < ed_video and ed_image < ed_erp and ed_image < ed_vp:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    elif ed_video < ed_image and ed_video < ed_erp and ed_video < ed_vp:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    elif ed_erp < ed_image and ed_erp < ed_video and ed_erp < ed_vp:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_erps -= 1
                        ed = ed_erp
                    else:
                        if input_ids[ed_vp] == image_token_id:
                            vp_type = 'image'
                        elif input_ids[ed_vp] == video_token_id:
                            vp_type = 'video'
                        else:
                            assert input_ids[ed_vp] == self.erp_context_token_id
                            vp_type = 'erp'
                        t, h, w = (
                            vp_grid_thw[vp_index][0],
                            vp_grid_thw[vp_index][1],
                            vp_grid_thw[vp_index][2],
                        )
                        second_per_grid_t = vp_second_per_grid_ts[vp_index]
                        prompt_mask = prompt_masks[vp_index]
                        vp_index += 1
                        remain_vps -= 1
                        ed = ed_vp

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    ## normalize type, send to device.
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.mllm.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    if input_ids[ed] == self.erp_context_token_id:
                        L = text_len + st_idx - 1
                        h_index, w_index = self.get_erp_rope_idx(L, llm_grid_h, llm_grid_w, llm_grid_t)
                        rope_index_3d = torch.stack([t_index, h_index, w_index])
                    else:
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        rope_index_3d = torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    assert rope_index_3d.shape[1] == llm_grid_t*llm_grid_h*llm_grid_w, f"rope_index_3d.shape: {rope_index_3d.shape}, llm_grid_t*llm_grid_h*llm_grid_w: {llm_grid_t*llm_grid_h*llm_grid_w}"
                    if vp_type == 'none':
                        llm_pos_ids_list.append(rope_index_3d)
                        st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                    else:
                        prompt_mask_flatten = prompt_mask.flatten().to(rope_index_3d.device)
                        vp_rope_index_3d = rope_index_3d[:, prompt_mask_flatten]
                        llm_pos_ids_list.append(vp_rope_index_3d)
                        st = ed + prompt_mask_flatten.sum().cpu().item()
                    
                    # st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                
                # print("attention_mask.sum(): ", attention_mask[i].sum())
                # print("st: ", st)

                if st < len(input_tokens):
                    # print("st < len(input_tokens)!!!!")
                    while len(llm_pos_ids_list) > 0 and llm_pos_ids_list[-1].numel()==0:
                        llm_pos_ids_list = llm_pos_ids_list[:-1]
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                # print("position_ids.shape: ", position_ids.shape)
                # print("llm_positions.shape: ", llm_positions.shape)
                # print("attention_mask.shape: ", attention_mask.shape)
                # exit(0)
                len_valid = attention_mask[i].sum()
                try:
                    position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device).to(position_ids.dtype)[:, :len_valid]
                except:
                    st = llm_positions.shape[1]
                    if st < len(input_tokens):
                        # print("st < len(input_tokens)!!!!")
                        while len(llm_pos_ids_list) > 0 and llm_pos_ids_list[-1].numel()==0:
                            llm_pos_ids_list = llm_pos_ids_list[:-1]
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        text_len = len(input_tokens) - st
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                    len_valid = attention_mask[i].sum()
                    position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device).to(position_ids.dtype)[:, :len_valid]
                    # print("position_ids.shape: ", position_ids.shape)
                    # print("llm_positions.shape: ", llm_positions.shape)
                    # print("len_valid: ", len_valid)
                    # exit(0)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas
    
    def _get_region_infos(self, masks, size):
        masks = F.interpolate(
            masks.unsqueeze(0),
            size=size,
            mode='nearest').squeeze(0)
        region_pixels = []
        for mask in masks:
            region_pixels.append(mask.bool().to(torch.int64).sum())
        return masks, region_pixels
    
    def predict_forward(
        self,
        image=None,
        question=None,
        mask_prompts=None,
        past_conversation=None,
        tokenizer=None,
        preprocessor=None,
        ori_image_size=None,
        **generation_kwargs,
    ):
        if '<erp>' in question:
            IMG_CONTEXT_TOKEN = '<|erp_pad|>'
            IMG_START_TOKEN = '<|vision_start|>'
            IMG_END_TOKEN = '<|vision_end|>'
            image_type = '<erp>'
        else:
            IMG_CONTEXT_TOKEN = '<|image_pad|>'
            IMG_START_TOKEN = '<|vision_start|>'
            IMG_END_TOKEN = '<|vision_end|>'
            image_type = '<image>'
        VP_START_TOKEN = '<vp>'
        VP_END_TOKEN = '</vp>'
        system_message = "You are a helpful assistant."
        self.image_context_token_id = self.mllm.model.config.image_token_id
        self.erp_context_token_id = tokenizer.convert_tokens_to_ids('<|erp_pad|>')
        self.cls_token_id = tokenizer.convert_tokens_to_ids('[CLS]')
        self.vp_start_token_id = tokenizer.convert_tokens_to_ids('<vp>')

        if past_conversation is None:
            past_conversation = [{"role": "system", "content": system_message}, ]
        
        if image is not None:
            image_processor = copy.deepcopy(preprocessor.image_processor)
            data_dict = image_processor.preprocess(image, return_tensors="pt")
            merge_length = image_processor.merge_size ** 2
            num_image_tokens = data_dict['image_grid_thw'][0].prod() // merge_length
            image_token_str = f'{IMG_START_TOKEN}' \
                f'{IMG_CONTEXT_TOKEN * num_image_tokens}' \
                f'{IMG_END_TOKEN}'
            question = question.replace(image_type, image_token_str)
            pixel_values = data_dict['pixel_values']
            image_grid_thw = data_dict['image_grid_thw']
            second_per_grid_t = 0
            num_frames = 1

            w, h = image.size
            if w > h:
                target_size = (1024, int(h/w*1024))
            else:
                target_size = (int(w/h*1024), 1024)
            resized_image = image.resize(target_size)
            cur_w, cur_h = resized_image.size
            padded_image = np.ones(shape=(1024, 1024, 3), dtype=np.uint8) * 255
            padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
            g_image = self.extra_image_processor.apply_image(padded_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            extra_pixel_values = [g_pixel_values]
            g_pixel_values = torch.stack([
                self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
            ])
            
            if mask_prompts is not None:
                prompt_mask_size = (data_dict['image_grid_thw'][0, 1]//image_processor.merge_size, data_dict['image_grid_thw'][0, 2]//image_processor.merge_size)
                if isinstance(mask_prompts, List):
                    if isinstance(mask_prompts, np.ndarray):
                        mask_prompts = np.stack(mask_prompts, axis=0)
                        mask_prompts_tensor = torch.from_numpy(mask_prompts)
                    elif isinstance(mask_prompts, torch.Tensor):
                        mask_prompts_tensor = torch.stack(mask_prompts, dim=0)
                    else:
                        raise NotImplementedError
                elif isinstance(mask_prompts, np.ndarray):
                    mask_prompts_tensor = torch.from_numpy(mask_prompts)
                elif isinstance(mask_prompts, torch.Tensor):
                    pass
                else:
                    raise NotImplementedError
                mask_prompts_tensor, region_pixels = self._get_region_infos(mask_prompts_tensor, prompt_mask_size)
                n_regions = len(mask_prompts_tensor)
                for _ in range(n_regions):
                    question = question.replace('<region>', VP_START_TOKEN + IMG_CONTEXT_TOKEN * region_pixels[_] + VP_END_TOKEN, 1)
                
                vp_second_per_grid_ts = [0] * len(mask_prompts_tensor)
                vp_grid_thw = [data_dict['image_grid_thw']]*len(mask_prompts_tensor)
            else:
                mask_prompts_tensor = None
                region_pixels = None
                vp_grid_thw = None
                vp_second_per_grid_ts = None
        else:
            pixel_values = None
            image_grid_thw = None
            second_per_grid_t = 0
            mask_prompts_tensor = None
            region_pixels = None
            vp_grid_thw = None
            vp_second_per_grid_ts = None
        
        past_conversation.append({"role": "user", "content": question})

        # tokenizer_copy = copy.deepcopy(tokenizer)
        # chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        # tokenizer_copy.chat_template = chat_template
        # input_id += tokenizer.apply_chat_template(
        #     [{"role": "system", "content": system_message}]
        # )

        text = preprocessor.apply_chat_template(
            past_conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = preprocessor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs['input_ids'].to(self.mllm.device)

        inputs_embeds = self.mllm.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.mllm.visual.device).to(self.mllm.visual.dtype)
            image_grid_thw = image_grid_thw.to(self.mllm.visual.device)
            image_embeds = self.mllm.visual(pixel_values, grid_thw=image_grid_thw)
            all_image_embeds = [image_embeds, ]
            if mask_prompts_tensor is not None:
                for vp_idx in range(len(mask_prompts_tensor)):
                    prompt_mask = mask_prompts_tensor[vp_idx]
                    prompt_mask = prompt_mask.flatten()
                    all_image_embeds.append(image_embeds[prompt_mask])
            image_embeds = torch.cat(all_image_embeds)

            mask1 = input_ids == self.image_context_token_id
            mask2 = input_ids == self.erp_context_token_id
            mask = mask1 | mask2
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        if vp_grid_thw is not None:
            vp_grid_thw = torch.cat(vp_grid_thw, dim=0)
        position_ids, rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            None,
            None,
            vp_grid_thw,
            vp_second_per_grid_ts,
            mask_prompts_tensor,
        )
        self.mllm.rope_deltas = rope_deltas

        generate_output = self.mllm.generate(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **generation_kwargs,
        )

        predict = tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=False).strip()
        output_text = predict

        past_conversation.append([{"role": "assistant", "content": output_text}])

        if image is None:
            return {'response': output_text, 'past_conversation': past_conversation}
        
        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)

        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.cls_token_id
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

        ret_masks = []
        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            pred_masks = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
            masks_1024 = F.interpolate(pred_masks, size=(1024, 1024), mode='bilinear', align_corners=False)
            depadded_masks = masks_1024[:, :, :target_size[1], :target_size[0]]

            if ori_image_size is None:
                return None
            w, h = ori_image_size
            masks = F.interpolate(depadded_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            masks = masks.sigmoid() > 0.5
            masks = masks.cpu().numpy()
            ret_masks.append(masks)

        return {'response': output_text, 'past_conversation': past_conversation, 'grounding_masks': ret_masks}


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]






        