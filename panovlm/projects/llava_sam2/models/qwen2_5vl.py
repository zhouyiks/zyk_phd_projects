import os
from typing import List, Optional, Tuple, Union
from types import MethodType

from mmengine import print_log
from mmengine.model import BaseModel
from mmengine.config import Config, ConfigDict

from xtuner.registry import BUILDER
from xtuner.model.utils import (find_all_linear_names, guess_load_checkpoint)
from peft import get_peft_model, prepare_model_for_kbit_training

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast


def prepare_inputs_for_generation_cache(
        self,
        input_ids,  ## yes
        past_key_values=None,
        attention_mask=None, ## yes
        inputs_embeds=None,
        cache_position=None,
        position_ids=None, 
        use_cache=True, 
        pixel_values=None, ## yes
        pixel_values_videos=None,
        image_grid_thw=None, ## yes
        video_grid_thw=None,
        second_per_grid_ts=None,
        visual_prompt_embeds_out=None, ## chenshihao
        img_context_token_idx=None, ## chenshihao
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        vit_embeds = self.visual.foward_add_vp_mutil_single(pixel_values, grid_thw=image_grid_thw, visual_prompt_embeds=visual_prompt_embeds_out)
        B, N = input_ids.shape
        input_embeds = self.model.get_input_embeddings()(input_ids).clone()

        B, N, C  = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)

        skip_this_case = False
        selected = (input_ids == img_context_token_idx)
        true_count = selected.sum().item()
        # print(f"Number of True elements: {true_count}")
        # print("vit_embeds:",vit_embeds.shape)
        input_ids = input_ids.reshape(B, N)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f"warning: {e}, input_embeds[selected].shape="
                  f"{input_embeds[selected].shape}, "
                  f"vit_embeds.shape={vit_embeds.shape}")
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
        input_embeds = input_embeds.reshape(B, N, C)


        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
            }
        )
        return model_inputs

def Qwen2_5_VLModel_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2_5VL_Slowfast(BaseModel):
    def __init__(self,
                 model_path,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 quantization_vit=False,
                 quantization_llm=False,
                 pretrained_pth=None,
                 special_tokens=None,
                 model_split=False,
                 ):
        print_log('Start to load InternVL_V1_5 model.', logger='current')
        super().__init__()

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        if quantization_vit:
            assert visual_encoder_lora is not None
        if quantization_llm:
            assert quantization_llm and llm_lora is not None
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if self.training:
            self.model.model.forward = MethodType(Qwen2_5_VLModel_forward, self.model.model)
        self.processor = AutoProcessor.from_pretrained(model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=8192,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer = tokenizer

        if special_tokens is not None:
            self._add_special_tokens(special_tokens)
        
        if self.freeze_llm:
            self.model.model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.visual.requires_grad_(False)

        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()
        
        if self.use_llm_lora:
            self.model.model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_cache, self.model.model)
            self._prepare_llm_for_lora(llm_lora)

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self._count = 0
        print_log(self, logger='current')
        print_log('Qwen2_5VL construction is complete', logger='current')

        self.transfer_to_hf = False
    
    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
            lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model.model = prepare_model_for_kbit_training(self.model.model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.model)
            lora_config.target_modules = modules
        
        self.model.model = get_peft_model(self.model.model, lora_config)
    
    def _prepare_visual_encoder_for_lora(self, lora_config):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.visual)
            lora_config.target_modules = modules
        self.model.visual = get_peft_model(self.model.visual, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.model.model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.model.model.gradient_checkpointing_disable()
    
    def _add_special_tokens(self, special_tokens):
        num_new_tokens = self.tokenizer.add_tokens(
            special_tokens, special_tokens=True)
        
        # preprocessor
        num_new_tokens = self.processor.tokenizer.add_tokens(special_tokens, special_tokens=True)

        if num_new_tokens > 0:
            self.model.model.resize_token_embeddings(len(self.tokenizer))
        
        self.vp_start_token_id = self.tokenizer.convert_tokens_to_ids('<vp>')

        self.erp_context_token_id = self.tokenizer.convert_tokens_to_ids('<|erp_pad|>')
        self.image_context_token_id = self.model.config.image_token_id
        self.video_context_token_id = self.model.config.video_token_id

    
    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values']
        image_grid_thw = data['image_grid_thw']

        input_ids = data['input_ids']
        # position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        image_flags = data['image_flags']
        query_embeds = data['query_embeds'] if 'query_embeds' in data else None
        prompt_masks = data['prompt_masks'] if 'prompt_masks' in data else None
        vp_gird_thw = data['vp_grid_thw'] if 'vp_grid_thw' in data else None
        vp_second_per_grid_ts = data['vp_second_per_grid_ts'] if 'vp_second_per_grid_ts' in data else None


        labels = data['labels']
        use_cache = False

        outputs = self._llm_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            image_flags=image_flags,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
            query_embeds=query_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            prompt_masks=prompt_masks,
            vp_grid_thw=vp_gird_thw,
            vp_second_per_grid_ts=vp_second_per_grid_ts,
        )

        return outputs
    
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
        spatial_merge_size = self.model.config.vision_config.spatial_merge_size
        image_token_id = self.model.config.image_token_id
        video_token_id = self.model.config.video_token_id
        vision_start_token_id = self.model.config.vision_start_token_id
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

                    try:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    except:
                        print(llm_pos_ids_list)
                        exit(0)
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    ## normalize type, send to device.
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.model.config.vision_config.tokens_per_second

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
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device).to(position_ids.dtype)[:, :len_valid]
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
        
    def _llm_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        query_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        vp_grid_thw: Optional[torch.LongTensor] = None,
        vp_second_per_grid_ts: Optional[torch.Tensor] = None,
        prompt_masks: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        vp_token_counts = (input_ids == self.vp_start_token_id).sum(dim=1)
        vision_input_counts = (input_ids == self.model.config.vision_start_token_id).sum(dim=1)

        # extract visual tokens
        vp_index = 0
        remain_vps = len(prompt_masks) if prompt_masks is not None else 0
        image_embeds = []
        for idx, (_pixel_values, _image_grid_thw) in enumerate(zip(pixel_values, image_grid_thw)):
            if vision_input_counts[idx] == 0:
                continue
            _image_embeds = self.model.visual(
                _pixel_values.to(self.model.visual.dtype),
                grid_thw=_image_grid_thw)
            image_embeds.append(_image_embeds)

            if vp_token_counts[idx] == 0 or remain_vps==0:
                continue
            for _ in range(vp_token_counts[idx]):
                prompt_mask = prompt_masks[vp_index]
                prompt_mask = prompt_mask.flatten()
                image_embeds.append(_image_embeds[prompt_mask])
                vp_index += 1
                remain_vps -= 1

        if len(image_embeds) > 0:
            image_embeds = torch.cat(image_embeds, dim=0)
            
            #TODO: support video inputs
            mask1 = input_ids == self.image_context_token_id
            mask2 = input_ids == self.erp_context_token_id
            mask = mask1 | mask2
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        image_grid_thw = [image_grid_thw_i for bs_i, image_grid_thw_i in enumerate(image_grid_thw) if vision_input_counts[bs_i]>0] if image_grid_thw is not None else None
        image_grid_thw = torch.cat(image_grid_thw) if len(image_grid_thw)>0 else None
        vp_grid_thw = torch.cat(vp_grid_thw) if vp_grid_thw is not None else None
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.model.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                    vp_grid_thw,
                    vp_second_per_grid_ts,
                    prompt_masks,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        outputs = self.model.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.model.rope_deltas
        )


        

