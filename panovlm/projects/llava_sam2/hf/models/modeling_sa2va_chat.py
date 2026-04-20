# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import Any, List, Optional, Tuple, Union, Dict, Set
from PIL import Image
import re

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torch.utils.checkpoint
import transformers

from .modeling_internlm2 import InternLM2ForCausalLM
from .modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging, TensorType
from transformers import StoppingCriteriaList, StoppingCriteria
from transformers.models.mask2former.image_processing_mask2former import (
    remove_low_and_no_objects, check_segment_validity)

from .configuration_sa2va_chat import Sa2VAChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn

from .templates import PROMPT_TEMPLATE

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

from types import MethodType
import torch.nn.functional as F

from transformers import Mask2FormerForUniversalSegmentation

from .mask2former import (
    Mask2FormerMaskedAttentionDecoder_forward_first3layers,
    Mask2FormerMaskedAttentionDecoder_forward_last3layers,
    Mask2FormerTransformerModule_forward_first_part,
    Mask2FormerTransformerModule_forward_second_part,
    Mask2FormerModel_forward_first_part,
    Mask2FormerModel_forward_second_part,
    Mask2FormerForUniversalSegmentation_forward_first_part,
    Mask2FormerForUniversalSegmentation_forward_second_part,
    _post_init,
    ov_class_predictor,
    Mask2FormerLoss_loss_labels,
    Mask2FormerLoss_loss_masks,
    Mask2FormerLoss_sample_points_using_uncertainty,
    Mask2FormerHungarianMatcher_forward,
)

from .constants import (
    IMG_CONTEXT_TOKEN, OBJ_CONTEXT_TOKEN, SEG_TOKEN, CLS_TOKEN, BG_CLS_TOKEN, OBJ_START_TOKEN, OBJ_END_TOKEN)



try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False

logger = logging.get_logger(__name__)

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word

def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class Sa2VAChatModel(PreTrainedModel):
    config_class = Sa2VAChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer', 'Mask2FormerForUniversalSegmentation']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: Sa2VAChatConfig, vision_model=None, language_model=None, mask2former=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.template = self.template.replace('-', '_')
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = PROMPT_TEMPLATE[self.template]
        self.template = self.conv_template
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)
        
        # mask2former
        if mask2former is None:
            self.mask2former = Mask2FormerForUniversalSegmentation(config.m2f_config)
        else:
            self.mask2former = mask2former
        assert self.mask2former.config.num_queries == config.num_m2f_queries
        self.num_m2f_queries =config. num_m2f_queries
        self.num_m2f_proposals = config.num_m2f_proposals
        self.m2f_input_size = 1024

        # register functions
        self.mask2former._post_init = MethodType(_post_init, self.mask2former)
        self.mask2former.ov_class_predictor = MethodType(ov_class_predictor, self.mask2former)
        self.mask2former.criterion.loss_labels = MethodType(Mask2FormerLoss_loss_labels, self.mask2former.criterion)
        self.mask2former.criterion.loss_masks = MethodType(Mask2FormerLoss_loss_masks, self.mask2former.criterion)
        self.mask2former.criterion.sample_points_using_uncertainty = MethodType(
            Mask2FormerLoss_sample_points_using_uncertainty, self.mask2former.criterion)
        self.mask2former.forward_first_part = MethodType(Mask2FormerForUniversalSegmentation_forward_first_part, self.mask2former)
        self.mask2former.forward_second_part = MethodType(Mask2FormerForUniversalSegmentation_forward_second_part, self.mask2former)
        self.mask2former.model.Mask2FormerModel_forward_first_part = MethodType(
            Mask2FormerModel_forward_first_part, self.mask2former.model)
        self.mask2former.model.Mask2FormerModel_forward_second_part = MethodType(
            Mask2FormerModel_forward_second_part, self.mask2former.model)
        self.mask2former.model.transformer_module.Mask2FormerTransformerModule_forward_first_part = MethodType(
            Mask2FormerTransformerModule_forward_first_part, self.mask2former.model.transformer_module
        )
        self.mask2former.model.transformer_module.Mask2FormerTransformerModule_forward_second_part = MethodType(
            Mask2FormerTransformerModule_forward_second_part, self.mask2former.model.transformer_module
        )
        self.mask2former.model.transformer_module.decoder.Mask2FormerMaskedAttentionDecoder_forward_first3layers = MethodType(
            Mask2FormerMaskedAttentionDecoder_forward_first3layers, self.mask2former.model.transformer_module.decoder
        )
        self.mask2former.model.transformer_module.decoder.Mask2FormerMaskedAttentionDecoder_forward_last3layers = MethodType(
            Mask2FormerMaskedAttentionDecoder_forward_last3layers, self.mask2former.model.transformer_module.decoder
        )
        self.mask2former.criterion.matcher.forward = MethodType(Mask2FormerHungarianMatcher_forward, self.mask2former.criterion.matcher)
        
        # post_init of mask2former
        self.mask2former._post_init()

        out_dim = config.m2f_config.hidden_dim
        in_dim = config.llm_config.hidden_size

        self.m2f_to_llm = nn.Sequential(
            nn.LayerNorm(out_dim,),
            nn.Linear(out_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim)
        )

        self.llm_to_m2f = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim * 2)
        )

        self.llm_to_cls = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

        self.init_prediction_config = False

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)
        else:
            raise NotImplementedError()

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data['labels']
        use_cache = False

        if 'vp_overall_mask' not in data.keys():
            vp_overall_mask = None
        else:
            vp_overall_mask = data['vp_overall_mask']

        if 'prompt_masks' in data.keys():
            prompt_masks = data['prompt_masks']
        else:
            prompt_masks = None

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
            vp_overall_mask=vp_overall_mask,
            prompt_masks=prompt_masks,
        )

        return outputs

    def _llm_forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            vp_overall_mask=None,
            prompt_masks=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.language_model.get_input_embeddings()(
            input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
        fast_vit_embeds = None

        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        self._count += 1

        if vp_overall_mask is not None and prompt_masks is not None:
            vp_embeds = []
            vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
            prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

            vp_overall_mask = vp_overall_mask[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

            i_vp_img = 0
            for i_img in range(len(vit_embeds)):
                vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                if vp_overall_mask[i_img]:
                    tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                    objects_prompt_masks = prompt_masks[i_vp_img]
                    n_obj = len(objects_prompt_masks)
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                    objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                    vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                    i_vp_img += 1
            vp_embeds = torch.cat(vp_embeds, dim=0)
        else:
            vp_embeds = None

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        if vp_embeds is None:
            try:
                input_embeds[selected] = vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vit_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vit_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vit_embeds) + 1
                    vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vit_embeds[:n_token]
        else:
            try:
                input_embeds[selected] = vp_embeds.reshape(-1, C)
            except Exception as e:
                vp_embeds = vp_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vp_embeds.shape={vp_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vp_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vp_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vp_embeds) + 1
                    vp_embeds = torch.cat([vp_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vp_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            prompt_masks=None,
            vp_overall_mask=None,
            query_embeds=None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.device
        assert self.img_context_token_id is not None

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                if type(pixel_values) is list or pixel_values.ndim == 5:
                    if type(pixel_values) is list:
                        pixel_values = [
                            x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                        ]
                    # b*n, c, h, w
                    pixel_values = torch.cat(
                        [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)

                vit_embeds = self.extract_feature(pixel_values.to(device))
            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            # object queries
            query_embeds = query_embeds.to(input_embeds.dtype)
            selected = (input_ids == self.obj_context_token_id)
            input_embeds[selected] = query_embeds.reshape(-1, C)
            
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def preparing_for_generation(self, tokenizer, max_new_tokens=2048, torch_dtype=torch.bfloat16):
        # set stop criteria and generation configs for model
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = tokenizer
        self.bot_name = 'BOT'
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )

        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True
        self.torch_dtype = torch_dtype
        self.to(torch_dtype)
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_size = patch_size

        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        # change phi3 prepare for generation fuction
        if self.config.llm_config.architectures[0] == 'Phi3ForCausalLM':
            self.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_phi3, self.language_model)
        
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        obj_context_token_id = tokenizer.convert_tokens_to_ids(OBJ_CONTEXT_TOKEN)
        self.obj_context_token_id = obj_context_token_id
        
        self.PROPOSAL_TOKENS = [SEG_TOKEN.format(id=str(i).zfill(3)) for i in range(self.num_m2f_proposals)]
        self.the_first_seg_token_idx = self.tokenizer(self.PROPOSAL_TOKENS[0], add_special_tokens=False).input_ids[0]
        self.the_last_seg_token_idx = self.tokenizer(self.PROPOSAL_TOKENS[-1], add_special_tokens=False).input_ids[0]
        self.cls_token_idx = self.tokenizer(CLS_TOKEN, add_special_tokens=False).input_ids[0]
        self.bg_cls_token_idx = self.tokenizer(BG_CLS_TOKEN, add_special_tokens=False).input_ids[0]

        return

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            m2f_processor=None,
    ):
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)

        if image is None and video is None and '<image>' not in past_text:
            text = text.replace('<image>', "")
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': None,
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': None,
                'vp_overall_mask': None,
                'm2f_inputs': None,
            }
        else:
            input_dict = {}
            if video is not None:
                pixel_values = []
                ori_image_size = video[0].size
                for frame_idx, frame_image in enumerate(video):
                    assert ori_image_size == frame_image.size
                    img = self.transformer(frame_image)
                    pixel_values.append(img)

                pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype)  # (n_f, 3, h, w)
                num_image_tokens = self.patch_token
                num_frames = len(pixel_values)

                # prepapre mask2former inputs
                m2f_pixel_values, m2f_pixel_masks = [], []
                for frame_idx, frame_image in enumerate(video):
                    assert ori_image_size == frame_image.size
                    w, h = frame_image.size
                    if w > h:
                        target_size = (self.m2f_input_size, int(h/w*self.m2f_input_size))
                    else:
                        target_size = (int(w/h*self.m2f_input_size), self.m2f_input_size)
                    
                    resized_frame_image = frame_image.resize(target_size)
                    cur_w, cur_h = resized_frame_image.size
                    padded_frame_image = np.ones(shape=(self.m2f_input_size, self.m2f_input_size, 3), dtype=np.uint8) * 255
                    padded_frame_image[:cur_h, :cur_w, :] = np.array(resized_frame_image)
                    m2f_inputs_i = m2f_processor(images=Image.fromarray(padded_frame_image), return_tensors="pt", do_resize=False)
                    m2f_pixel_values.append(m2f_inputs_i['pixel_values'])
                    m2f_pixel_masks.append(m2f_inputs_i['pixel_mask'])
                m2f_inputs = {
                    'pixel_values': torch.cat(m2f_pixel_values, dim=0),
                    'pixel_mask': torch.cat(m2f_pixel_masks, dim=0)}
            else:
                ori_image_size = image.size

                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                            self.max_dynamic_patch,
                                            self.image_size, self.use_thumbnail)

                pixel_values = [self.transformer(patch) for patch in images]
                pixel_values = torch.stack(pixel_values).to(self.torch_dtype)
                num_image_tokens = pixel_values.shape[0] * self.patch_token
                num_frames = 1

                w, h = image.size
                if w > h:
                    target_size = (self.m2f_input_size, int(h/w*self.m2f_input_size))
                else:
                    target_size = (int(w/h*self.m2f_input_size), self.m2f_input_size)
                
                resized_image = image.resize(target_size)
                cur_w, cur_h = resized_image.size
                padded_image = np.ones(shape=(self.m2f_input_size, self.m2f_input_size, 3), dtype=np.uint8) * 255
                padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
                m2f_inputs = m2f_processor(images=Image.fromarray(padded_image), return_tensors="pt", do_resize=False)

            input_dict['pixel_values'] = pixel_values
            
            #TODO add a frame tag to indicate the order
            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            object_token_str = f"{OBJ_START_TOKEN}"\
                               f"{OBJ_CONTEXT_TOKEN * self.num_m2f_queries}"\
                               f"{OBJ_END_TOKEN}"
            image_token_str = image_token_str + '\n' + object_token_str + '\n'
            image_token_str = image_token_str * num_frames
            image_token_str = image_token_str.strip()

            if '<image>' in text or mask_prompts is not None:
                assert past_text is None or len(past_text) == 0
            text = text.replace('<image>', image_token_str)
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            # encode multi-scale visual features into 100~300 queries
            m2f_inputs['pixel_values'] = m2f_inputs['pixel_values'].to(self.mask2former.dtype).to(self.mask2former.device)
            m2f_inputs['pixel_mask'] = m2f_inputs['pixel_mask'].to(self.mask2former.dtype).to(self.mask2former.device)
            query_features, pixel_level_module_output = \
                self.mask2former.forward_first_part(**m2f_inputs)
            query_embeds = self.m2f_to_llm(query_features)  # BS, m2f_NQ, 2048


            mm_inputs = {
                'pixel_values': input_dict['pixel_values'],
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'query_embeds': query_embeds,
                # 'prompt_masks': mask_prompts,
                # 'vp_overall_mask': input_dict['vp_overall_mask'],
            }

        generate_output = self.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        predict = self.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=False).strip()
        
        ret_masks = []
        if image is None and video is None and '<image>' not in past_text:
            return {'prediction': predict, 'prediction_masks': ret_masks, 'm2f_outputs': None}

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)

        # get cls tokens
        bg_cls_token_id = torch.as_tensor([self.bg_cls_token_idx,], dtype=ids.dtype, device=ids.device)
        bg_cls_embedding = self.language_model.get_input_embeddings()(bg_cls_token_id).clone()
        output_ids = generate_output.sequences[0][:-1]
        cls_token_mask = ids[0] == self.cls_token_idx

        # get seg tokens
        seg_token_mask = (output_ids >= self.the_first_seg_token_idx) & (output_ids <= self.the_last_seg_token_idx)

        do_pano_seg = torch.any(cls_token_mask) & torch.any(seg_token_mask)

        reason_cls_token_mask = output_ids == self.cls_token_idx

        do_reason_seg = torch.any(reason_cls_token_mask) & torch.any(seg_token_mask)

        if not do_pano_seg and not do_reason_seg:
            return {'prediction': predict, 'prediction_masks': ret_masks, 'm2f_outputs': None}
        
        # get seg tokens
        seg_hidden_states = last_hidden_states[-len(seg_token_mask):][seg_token_mask].unsqueeze(0)
        seg_hidden_states = self.llm_to_m2f(seg_hidden_states)
        
        if do_pano_seg:
            cls_hidden_states = last_hidden_states[:len(cls_token_mask)][cls_token_mask]
            text_classifier = self.llm_to_cls(torch.cat([cls_hidden_states, bg_cls_embedding], dim=0))
            seg_hidden_states = seg_hidden_states.transpose(0, 1)

            # proposals go through mask2former decoder layers
            m2f_outputs = self.mask2former.forward_second_part(
                query_features=seg_hidden_states[:, :, :self.mask2former.config.hidden_dim], # q, b, c
                query_embeddings=seg_hidden_states[:, :, self.mask2former.config.hidden_dim:], # q, b, c
                pixel_level_module_output=pixel_level_module_output,
                text_classifier=[text_classifier, ],
                mask_labels=None,
                class_labels=None,
                **m2f_inputs
            )

            tags = re.findall(r'<p>(.*?)</p>', input_text)
            label_id_to_text = {id: tag for id, tag in enumerate(tags)}
            
            class_queries_logits = m2f_outputs.class_queries_logits
            masks_queries_logits = m2f_outputs.masks_queries_logits

            m2f_masks = {'label_id_to_text': label_id_to_text,
                        'class_queries_logits': class_queries_logits,
                        'masks_queries_logits': masks_queries_logits}

            return {'prediction': predict, 'prediction_masks': ret_masks, 'm2f_outputs': m2f_masks}
        elif do_reason_seg:
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    def post_process_panoptic_segmentation(
        self,
        class_queries_logits,
        masks_queries_logits,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: Optional[Set[int]] = None,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict]:
        
        if label_ids_to_fuse is None:
            logger.warning("`label_ids_to_fuse` unset. No instance will be fused.")
            label_ids_to_fuse = set()

        batch_size = len(class_queries_logits)

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(batch_size):
            height, width = target_sizes[i]
            long_edge = height if height > width else width
            masks_queries_logits_i = torch.nn.functional.interpolate(
                masks_queries_logits[i:i+1], size=(long_edge, long_edge), mode="bilinear", align_corners=False
            )
            
            mask_probs = masks_queries_logits_i[0].sigmoid()

            num_labels = class_queries_logits[i].shape[-1] - 1

            pred_scores, pred_labels = nn.functional.softmax(class_queries_logits[i], dim=-1).max(-1)

            mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(
                mask_probs, pred_scores, pred_labels, threshold, num_labels
            )

            results.append({'ids': pred_labels_item, 'masks': mask_probs_item > mask_threshold})
            
            # # No mask found
            # if mask_probs_item.shape[0] <= 0:
            #     segmentation = torch.zeros((height, width)) - 1
            #     results.append({"segmentation": segmentation, "segments_info": []})
            #     continue

            # # Get segmentation map and segment information of batch item
            # target_size = target_sizes[i] if target_sizes is not None else None
            # segmentation, segments = compute_segments(
            #     mask_probs=mask_probs_item,
            #     pred_scores=pred_scores_item,
            #     pred_labels=pred_labels_item,
            #     mask_threshold=mask_threshold,
            #     overlap_mask_area_threshold=overlap_mask_area_threshold,
            #     label_ids_to_fuse=label_ids_to_fuse,
            #     target_size=target_size,
            # )
            
            # results.append({"segmentation": segmentation, "segments_info": segments})

        return results

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation_phi3(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (past_key_values is None or len(past_key_values)==0):
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs


# Copied from transformers.models.detr.image_processing_detr.compute_segments
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[Set[int]] = None,
    target_size: Tuple[int, int] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: List[Dict] = []

    if target_size is not None:
        mask_probs = mask_probs[..., :height, :width]

    current_segment_id = 0

    # Weigh each mask by its prediction score
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # Keep track of instances of each class
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # Check if mask exists and large enough to be a segment
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            # Add current object segment to final segmentation map
            segmentation[mask_k] = current_segment_id
            segment_score = round(pred_scores[k].item(), 6)
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id

    return segmentation, segments