from typing import Literal, Optional, List, Tuple, Union, Dict, Set
import re
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print("use npu success!")
except:
    print("npu not enabled!")
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import numpy as np

from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict

from pycocotools import mask as _mask

from xtuner.model.utils import (
    find_all_linear_names,
    guess_load_checkpoint,
    make_inputs_require_grad,
)

from scipy.optimize import linear_sum_assignment

from mmengine.model import BaseModel
from mmengine.config import Config, ConfigDict


from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import auto_docstring, is_accelerate_available, logging, TensorType
from transformers import GenerationConfig
from transformers.models.mask2former.image_processing_mask2former import (
    remove_low_and_no_objects, check_segment_validity)
from transformers import StoppingCriteriaList, StoppingCriteria
from peft import get_peft_model, prepare_model_for_kbit_training

if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

from projects.llava_dino_m2f.models.backbone.dinov3_adapter import DINOv3_Adapter
from projects.llava_dino_m2f.models.heads.mask2former_head import Mask2FormerHead
from projects.llava_dino_m2f.models.backbone.backbones import dinov3_vitl16, dinov3_vit7b16

from projects.llava_dino_m2f.datasets.constants import SEG_TOKEN, CLS_TOKEN, BG_CLS_TOKEN, IMG_CONTEXT_TOKEN, OBJ_CONTEXT_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN, OBJ_START_TOKEN, OBJ_END_TOKEN

from projects.llava_dino_m2f.models.losses import CrossEntropyLoss, DiceLoss, point_sample, get_uncertain_point_coords_with_randomness
from projects.llava_dino_m2f.datasets.dynamic_preprocess import dynamic_preprocess
from projects.llava_dino_m2f.models.templates import PROMPT_TEMPLATE


BACKBONE_INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


class LLaVADINOM2F(BaseModel):
    def __init__(
        self,
        mllm,
        tokenizer,
        special_tokens=None,
        llm_lora=None,
        freeze_llm=False,
        backbone_name="dinov3_vitl16",
        backbone_weights=None,
        num_proposal_layers: int = 3,
        num_queries: int = 100,
        pretrained_pth: str = None,
        # mask loss
        loss_sample_points=False,
        num_points=12544,
        loss_mask=None,
        loss_dice=None,
        dinov3_input_mean=(0.485, 0.456, 0.406),
        dinov3_input_std=(0.229, 0.224, 0.225),
    ):
        super().__init__()

        self.PROPOSAL_TOKENS = [SEG_TOKEN.format(id=str(i).zfill(3)) for i in range(num_queries)]
        if special_tokens is None:
            special_tokens = [CLS_TOKEN, BG_CLS_TOKEN] + self.PROPOSAL_TOKENS
        self.special_tokens = special_tokens

        self.mllm = BUILDER.build(mllm)
        self.tokenizer = BUILDER.build(tokenizer)

        if self.special_tokens is not None:
            self._add_special_tokens()
        
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.mllm.img_context_token_id = img_context_token_id
        obj_context_token_id = self.tokenizer.convert_tokens_to_ids(OBJ_CONTEXT_TOKEN)
        self.mllm.obj_context_token_id = obj_context_token_id
        
        self.mllm.vision_model.requires_grad_(False)
        if freeze_llm:
            self.mllm.language_model.requires_grad_(False)
        if hasattr(self.mllm.language_model, 'enable_input_require_grads'):
            self.mllm.language_model.enable_input_require_grads()
        else:
            self.mllm.language_model.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
        self.freeze_llm = freeze_llm

        self.gradient_checkpointing_enable()

        self.use_llm_lora = False
        if llm_lora is not None:
            self._prepare_llm_for_lora(llm_lora)
            self.use_llm_lora = True
        self.mllm.language_model.model.model.norm.weight.requires_grad = True
        self.mllm.language_model.model.model.embed_tokens.weight.requires_grad = True
        self.mllm.language_model.model.lm_head.weight.requires_grad = True
        self.mllm.language_model.model.tie_weights()

        if backbone_name == 'dinov3_vitl16':
            dinov3_model = dinov3_vitl16(pretrained=True, weights=backbone_weights)
        elif backbone_name == 'dinov3_vit7b16':
            dinov3_model = dinov3_vit7b16(pretrained=True, weights=backbone_weights)
        else:
            raise NotImplementedError
        self.backbone_model = DINOv3_Adapter(
            dinov3_model,
            interaction_indexes=BACKBONE_INTERMEDIATE_LAYERS[backbone_name],
        )
        # self.backbone_model.eval()
        embed_dim = self.backbone_model.backbone.embed_dim
        patch_size = self.backbone_model.patch_size
        self.mask_decoder = Mask2FormerHead(
            input_shape={
                "1": [embed_dim, patch_size * 4, patch_size * 4, 4],
                "2": [embed_dim, patch_size * 2, patch_size * 2, 4],
                "3": [embed_dim, patch_size, patch_size, 4],
                "4": [embed_dim, int(patch_size / 2), int(patch_size / 2), 4],
            },
            hidden_dim=2048,
            ignore_value=255,
            num_proposal_layers=num_proposal_layers,
            num_queries=num_queries,
        )

        m2f_dim = 2048
        llm_dim = self.mllm.config.llm_config.hidden_size

        self.m2f_to_llm = nn.Sequential(
            nn.LayerNorm(m2f_dim,),
            nn.Linear(m2f_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

        self.llm_to_m2f = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, m2f_dim * 2),
            nn.GELU(),
            nn.Linear(m2f_dim * 2, m2f_dim * 2)
        )

        self.llm_to_cls = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, m2f_dim),
            nn.GELU(),
            nn.Linear(m2f_dim, m2f_dim)
        )

        # Convert modules to match mllm dtype
        self.m2f_to_llm.to(self.mllm.dtype)
        self.llm_to_m2f.to(self.mllm.dtype)
        self.llm_to_cls.to(self.mllm.dtype)
        self.loss_dice = BUILDER.build(loss_dice)

        self.weight_dict: dict[str, float] = {
            "loss_cross_entropy": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0,
        }
        self.criterion = Mask2FormerLoss(weight_dict=self.weight_dict)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f"Load pretrained weight from {pretrained_pth}")

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75
        self.num_queries = num_queries
        self.m2f_dim = m2f_dim
        self.llm_dim = llm_dim

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.dinov3_transformer = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), antialias=True),
            T.Normalize(
                mean=dinov3_input_mean,
                std=dinov3_input_std,
            ),
        ])

        self.template = PROMPT_TEMPLATE['internlm2_chat']

        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria
    
    def _add_special_tokens(self):
        num_new_tokens = self.tokenizer.add_tokens(
            self.special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.mllm.language_model.resize_token_embeddings(len(self.tokenizer))

        self.the_first_seg_token_idx = self.tokenizer(self.PROPOSAL_TOKENS[0], add_special_tokens=False).input_ids[0]
        self.the_last_seg_token_idx = self.tokenizer(self.PROPOSAL_TOKENS[-1], add_special_tokens=False).input_ids[0]
        self.cls_token_idx = self.tokenizer(CLS_TOKEN, add_special_tokens=False).input_ids[0]
        self.bg_cls_token_idx = self.tokenizer(BG_CLS_TOKEN, add_special_tokens=False).input_ids[0]

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.mllm.language_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.mllm.language_model.gradient_checkpointing_disable()

    def _parse_lora_config(self, lora_config):
        if (
            isinstance(lora_config, dict)
            or isinstance(lora_config, Config)
            or isinstance(lora_config, ConfigDict)
        ):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.mllm.language_model = prepare_model_for_kbit_training(
            self.mllm.language_model, use_activation_checkpointing
        )
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.mllm.language_model)
            lora_config.target_modules = modules
        self.mllm.language_model = get_peft_model(
            self.mllm.language_model, lora_config
        )
    
    def _merge_lora(self):
        try:
            self.mllm.language_model = self.mllm.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        return

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        loss_dict: dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict
    
    def state_dict(self, *args, **kwargs):
        state_dict = super(LLaVADINOM2F, self).state_dict(*args, **kwargs)
        from collections import OrderedDict

    
        to_return = OrderedDict()
        # Step 1. visual_encoder. By convention, we do not fine-tune the visual encoder.
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.mllm.language_model, state_dict=state_dict)
            )
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'language_model.' in k})
            # raise NotImplementedError
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mlp1.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.multi_modal_projector.' in k})
        
        # Step 4. Mask2Former
        to_return.update(
            {
                k: v
                for k, v in state_dict.items() if 'backbone_model.' in k or 'mask_decoder.' in k
            }
        )

        # Step 5. others (fcs)
        to_return.update(
            {
                k: v
                for k, v in state_dict.items() if 'm2f_to_llm.' in k
            }
        )
        to_return.update(
            {
                k: v
                for k, v in state_dict.items() if 'llm_to_m2f.' in k
            }
        )
        to_return.update(
            {
                k: v
                for k, v in state_dict.items() if 'llm_to_cls.' in k
            }
        )
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.norm.weight' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'lm_head.weight' in k})
        to_return.update(
            {k: v
                for k, v in state_dict.items() if 'embed_tokens.weight' in k})
        # print("===================================>>>>>>save state dict:\n")
        # print(to_return.keys())
        # exit(0)
        
        return to_return


    def forward(self, data, data_samples=None, mode='loss'):
        # print("===================================>>>>>>trainable params:\n")
        # for n, p in self.named_parameters():
        #     if p.requires_grad:
        #         print(n)
        # exit(0)
        dinov3_inputs = data.pop('dinov3_inputs')
        assert dinov3_inputs.shape[0] == 1
        dinov3_inputs = dinov3_inputs[0]
        aspect_ratio = data['aspect_ratio'] # 1, 2

        self.backbone_model.to(torch.float32)
        self.mask_decoder.to(torch.float32)
        dinov3_inputs = dinov3_inputs.to(self.backbone_model.norm1.weight.dtype)
        dinov3_outputs = self.backbone_model(dinov3_inputs)
        w, h = aspect_ratio[0]
        new_dinov3_outputs = {}
        for k, v in dinov3_outputs.items():
            N, C, H, W = v.shape
            assert N == h*w
            v = v.view(h, w, C, H, W)
            v = v.permute(0, 2, 3, 1, 4).contiguous()
            v = v.reshape(h, C, H, w*W).contiguous()
            v = v.permute(1, 0, 2, 3).contiguous()
            v = v.reshape(1, C, h*H, w*W).contiguous()
            new_dinov3_outputs[k] = v

        dinov3_mask_features, _, dinov3_multi_scale_features = self.mask_decoder.pixel_decoder.forward_features(new_dinov3_outputs)
        mask_proposals = self.mask_decoder.predictor.forward_pre(dinov3_multi_scale_features, dinov3_mask_features)
        mask_proposals = mask_proposals.to(self.mllm.dtype)
        mask_proposals = self.m2f_to_llm(mask_proposals)

        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.mllm.vision_model.dtype) for image in pixel_values], dim=0)
        else:
            raise NotImplementedError()

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        image_flags = data['image_flags']

        labels = data['labels']
        use_cache = False

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
            mask_proposals=mask_proposals,
        )

        # 1.1 extract proposals hidden
        gt_masks = data.pop('masks')
        gt_class_ids = data.pop('class_ids')
        if gt_masks is None:
            seg_valid = False
        else:
            seg_valid = True
        
        input_ids = data['input_ids']
        # seg_token_mask = (input_ids == self.mllm.obj_context_token_id)
        seg_token_mask = (input_ids >= self.the_first_seg_token_idx) & (input_ids <= self.the_last_seg_token_idx)
        hidden_states = outputs.hidden_states
        hidden_states = self.llm_to_m2f(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :self.num_queries].flatten(0, 1) + _zero
        
        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += self.num_queries
        
        pred_embeddings = pred_embeddings.reshape(-1, self.num_queries, self.m2f_dim*2)
        pred_embeddings = pred_embeddings.transpose(0, 1)

        # 1.2 extract [CLS] tokens
        bs = input_ids.shape[0]
        bg_cls_token_id = torch.as_tensor([self.bg_cls_token_idx,], dtype=input_ids.dtype, device=input_ids.device)
        bg_cls_embedding = self.mllm.language_model.get_input_embeddings()(bg_cls_token_id).clone()
        bg_cls_embedding = bg_cls_embedding.unsqueeze(0).repeat(bs, 1, 1)
        cls_token_mask = input_ids == self.cls_token_idx
        hidden_states = outputs.hidden_states
        hidden_states = self.llm_to_cls(torch.cat([hidden_states[-1], bg_cls_embedding], dim=1))
        bg_cls_embedding = hidden_states[:, -1:, :]
        hidden_states = hidden_states[:, :-1, :]
        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            text_classifier = hidden_states[cls_token_mask] + _zero
        else:
            text_classifier = hidden_states[:, :1].flatten(0, 1) + _zero

        cls_token_counts = cls_token_mask.int().sum(-1)
        if not seg_valid:
            cls_token_counts += 1
        
        text_classifier_list_ = torch.split(text_classifier, cls_token_counts.tolist(), dim=0)
        text_classifier_list = []
        for bs_i, item in enumerate(text_classifier_list_):
            if len(item) != 0:
                text_classifier_list.append(torch.cat([item, bg_cls_embedding[bs_i]], dim=0))
        
        # 2. proposals go through mask2former decoder layers
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            m2f_outputs = self.mask_decoder.predictor.forward_post(
                pred_embeddings[:, :, :self.m2f_dim],
                pred_embeddings[:, :, self.m2f_dim:],
                text_classifier_list,
                dinov3_multi_scale_features,
                dinov3_mask_features
            )

        loss_dict = self.get_loss_dict(
            masks_queries_logits=m2f_outputs['pred_masks'],
            class_queries_logits=m2f_outputs['pred_logits'],
            mask_labels=gt_masks,
            class_labels=gt_class_ids,
            auxiliary_predictions=m2f_outputs['aux_outputs'],
        )

        mask_loss = []
        class_loss = []
        for k, v in loss_dict.items():
            if 'mask' in k or 'dice' in k:
                mask_loss.append(v)
            else:
                class_loss.append(v)
        mask_loss = sum(mask_loss)
        class_loss = sum(class_loss)

        m2f_loss = sum(loss_dict.values())
        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        m2f_loss = m2f_loss * _scale

        loss_dict = {
            'llm_loss': outputs.loss,
            'mask_loss': mask_loss,
            'class_loss': class_loss,
        }
        return loss_dict
    
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
        mask_proposals: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.mllm.config.use_return_dict
        
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.mllm.language_model.get_input_embeddings()(
            input_ids).clone()
        
        vit_embeds = self.mllm.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)

        vit_embeds = vit_embeds[image_flags == 1]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B*N)
        selected = (input_ids == self.mllm.img_context_token_id)
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
        
        mask_proposals = mask_proposals.to(input_embeds.dtype)
        selected = (input_ids == self.mllm.obj_context_token_id)
        try:
            input_embeds[selected] = mask_proposals.reshape(-1, C)
        except Exception as e:
            mask_proposals = mask_proposals.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape='
                    f'{input_embeds[selected].shape}, '
                    f'mask_proposals.shape={mask_proposals.shape}')
            n_token = selected.sum()
            if n_token > len(mask_proposals):
                print(f"Wrong !!! {n_token} image tokens in text but only {len(mask_proposals)} query embeds !!!")
                expand_ratio = n_token // len(mask_proposals) + 1
                mask_proposals = torch.cat([mask_proposals] * expand_ratio, dim=0)

            input_embeds[selected] = mask_proposals[:n_token]
        
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.mllm.language_model(
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
                -1, self.mllm.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

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
        mask_proposals=None,
        **generate_kwargs,
    ):
        device = self.mllm.device
        assert self.mllm.img_context_token_id is not None

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
                        [image.to(self.mllm.vision_model.dtype) for image in pixel_values], dim=0)

                vit_embeds = self.mllm.extract_feature(pixel_values.to(device).to(self.mllm.dtype))
            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            input_embeds = self.mllm.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.mllm.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            # object queries
            mask_proposals = mask_proposals.to(input_embeds.dtype)
            selected = (input_ids == self.mllm.obj_context_token_id)
            input_embeds[selected] = mask_proposals.reshape(-1, C)
            
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.mllm.language_model.get_input_embeddings()(input_ids)

        outputs = self.mllm.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
    
    def chat(
        self,
        image=None,
        text=None,
        dinov3_min_dynamic_patch=1,
        dinov3_max_dynamic_patch=17,
        dinov3_input_size=224,
    ):
        ori_image_size = image.size

        images, _ = dynamic_preprocess(image, 1, 12, 448, True)
        pixel_values = [self.transformer(patch) for patch in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16)
        num_image_tokens = pixel_values.shape[0] * 256

        input_dict = {}
        input_dict['pixel_values'] = pixel_values
        image_token_str = f'{IMG_START_TOKEN}' \
                        f'{IMG_CONTEXT_TOKEN * num_image_tokens}' \
                        f'{IMG_END_TOKEN}'
        object_token_str = f"{OBJ_START_TOKEN}"\
                        f"{OBJ_CONTEXT_TOKEN * self.num_queries}"\
                        f"{OBJ_END_TOKEN}"
        image_token_str = image_token_str + '\n' + object_token_str + '\n'
        image_token_str = image_token_str.strip()

        text = text.replace('<image>', image_token_str)
        input_text = ''
        input_text += self.template['INSTRUCTION'].format(
            input=text, round=1, bot_name='BOT'
        )
        ids = self.tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        attention_mask = torch.ones_like(ids, dtype=torch.bool)

        images, aspect_ratio = dynamic_preprocess(image, dinov3_min_dynamic_patch,
                                    dinov3_max_dynamic_patch,
                                    dinov3_input_size, False)
        dinov3_pixel_values = [self.dinov3_transformer(patch) for patch in images]
        dinov3_pixel_values = torch.stack(dinov3_pixel_values)
   
        self.backbone_model.to(torch.float32)
        self.mask_decoder.to(torch.float32)
        dinov3_pixel_values = dinov3_pixel_values.to(self.backbone_model.norm1.weight.dtype).cuda()
        dinov3_outputs = self.backbone_model(dinov3_pixel_values)
        w, h = aspect_ratio
        new_dinov3_outputs = {}
        for k, v in dinov3_outputs.items():
            N, C, H, W = v.shape
            assert N == h*w
            v = v.view(h, w, C, H, W)
            v = v.permute(0, 2, 3, 1, 4).contiguous()
            v = v.reshape(h, C, H, w*W).contiguous()
            v = v.permute(1, 0, 2, 3).contiguous()
            v = v.reshape(1, C, h*H, w*W).contiguous()
            new_dinov3_outputs[k] = v

        dinov3_mask_features, _, dinov3_multi_scale_features = self.mask_decoder.pixel_decoder.forward_features(new_dinov3_outputs)
        mask_proposals = self.mask_decoder.predictor.forward_pre(dinov3_multi_scale_features, dinov3_mask_features)
        mask_proposals = mask_proposals.to(self.mllm.dtype)
        mask_proposals = self.m2f_to_llm(mask_proposals)

        mm_inputs = {
            'pixel_values': input_dict['pixel_values'],
            'input_ids': ids,
            'attention_mask': attention_mask,
            'position_ids': None,
            'past_key_values': None,
            'labels': None,
            'mask_proposals': mask_proposals,
        }

        default_generation_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )

        gen_config = GenerationConfig(**default_generation_kwargs)
        generate_output = self.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        predict = self.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=False).strip()
        
        if image is None:
            return {'response': predict, 'masks': None, 'tags': None}

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)

        # get cls tokens
        bg_cls_token_id = torch.as_tensor([self.bg_cls_token_idx,], dtype=ids.dtype, device=ids.device)
        bg_cls_embedding = self.mllm.language_model.get_input_embeddings()(bg_cls_token_id).clone()
        output_ids = generate_output.sequences[0][:-1]
        cls_token_mask = ids[0] == self.cls_token_idx

        # get seg tokens
        seg_token_mask = (output_ids >= self.the_first_seg_token_idx) & (output_ids <= self.the_last_seg_token_idx)
        # seg_token_mask = ids[0] == self.mllm.obj_context_token_id

        do_pano_seg = torch.any(cls_token_mask) & torch.any(seg_token_mask)

        if not do_pano_seg:
            return {'response': predict, 'masks': None, 'tags': None}
        
        # get seg tokens
        seg_hidden_states = last_hidden_states[-len(seg_token_mask):][seg_token_mask].unsqueeze(0)
        # seg_hidden_states = last_hidden_states[:len(seg_token_mask)][seg_token_mask].unsqueeze(0)
        seg_hidden_states = self.llm_to_m2f(seg_hidden_states)

        if do_pano_seg:
            cls_hidden_states = last_hidden_states[:len(cls_token_mask)][cls_token_mask]
            text_classifier = self.llm_to_cls(torch.cat([cls_hidden_states, bg_cls_embedding], dim=0))
            seg_hidden_states = seg_hidden_states.transpose(0, 1)

            # proposals go through mask2former decoder layers
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=seg_hidden_states.is_cuda):
                m2f_outputs = self.mask_decoder.predictor.forward_post(
                    seg_hidden_states[:, :, :self.m2f_dim],
                    seg_hidden_states[:, :, self.m2f_dim:],
                    [text_classifier],
                    dinov3_multi_scale_features,
                    dinov3_mask_features
                )

            tags = re.findall(r'<p>(.*?)</p>', input_text)
            label_id_to_text = {id: tag for id, tag in enumerate(tags)}
            
            masks_queries_logits=m2f_outputs['pred_masks'],
            class_queries_logits=m2f_outputs['pred_logits'],
            
            post_m2f_outputs = self.post_process_panoptic_segmentation(
                class_queries_logits[0],
                masks_queries_logits[0],
                target_sizes=[(ori_image_size[1], ori_image_size[0])]
            )
            # segmentation = post_m2f_outputs[0]['segmentation']
            # segments_info = post_m2f_outputs[0]['segments_info']

            # pano_masks, pano_tags = [], []
            # for item in segments_info:
            #     mask = segmentation == item['id']
            #     pano_masks.append(mask.unsqueeze(0).cpu().numpy())
            #     pano_tags.append(label_id_to_text[item['label_id']])

            # pano_masks = np.concatenate(pano_masks, axis=0)

            masks = post_m2f_outputs[0]['masks']
            ids = post_m2f_outputs[0]['ids']
            pano_masks, pano_tags = [], []
            for id, mask in zip(ids, masks):
                pano_masks.append(mask.unsqueeze(0).cpu().numpy())
                pano_tags.append(label_id_to_text[id.item()])
            pano_masks = np.concatenate(pano_masks, axis=0)
            
            return {'response': predict, 'masks': pano_masks, 'tags': pano_tags}
        else:
            raise NotImplementedError
        
    def post_process_panoptic_segmentation(
        self,
        class_queries_logits,
        masks_queries_logits,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse=None,
        target_sizes=None,
    ):
        if label_ids_to_fuse is None:
            label_ids_to_fuse = set()

        batch_size = len(class_queries_logits)

        print("masks_queries_logits.shape: ", masks_queries_logits.shape)

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


# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
class Mask2FormerLoss(nn.Module):
    def __init__(self, weight_dict: dict[str, float]):
        """
        The Mask2Former Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we
        compute hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair
        of matched ground-truth / prediction (supervise class and mask)

        Args:
            config (`Mask2FormerConfig`):
                The configuration for Mask2Former model also containing loss calculation specific parameters.
            weight_dict (`dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
        """
        super().__init__()
        self.num_labels = 80
        self.weight_dict = weight_dict

        # Weight to apply to the null class
        self.eos_coef = 0.1
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = 12544
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.matcher = Mask2FormerHungarianMatcher(
            cost_class=2.0,
            cost_dice=5.0,
            cost_mask=5.0,
            num_points=self.num_points,
        )

    def _max_by_axis(self, sizes: list[list[int]]) -> list[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    # Adapted from nested_tensor_from_tensor_list() in original implementation
    def _pad_images_to_max_in_batch(self, tensors: list[Tensor]) -> tuple[Tensor, Tensor]:
        # get the maximum size in the batch
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        # compute final size
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # pad the tensors to the size of the biggest one
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks

    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: list[Tensor], indices: tuple[np.array]
    ) -> dict[str, Tensor]:
        batch_size = len(class_queries_logits)
        num_queries = class_queries_logits[0].shape[0]
        all_ce_loss = []
        for i in range(batch_size):
            num_labels_plus1 = class_queries_logits[i].shape[-1]
            empty_weight = torch.ones(num_labels_plus1)
            empty_weight[-1] = self.eos_coef
            empty_weight = empty_weight.to(class_queries_logits[i].device).to(class_queries_logits[i].dtype)
            criterion = nn.CrossEntropyLoss(weight=empty_weight, reduction='none')
            target_classes_o = class_labels[i][indices[i][1]]
            target_classes = torch.full(
                 (num_queries, ), fill_value=num_labels_plus1-1, dtype=torch.int64, device=class_queries_logits[i].device)
            target_classes[indices[i][0]] = target_classes_o.to(class_queries_logits[i].device)
            target_classes = target_classes.unsqueeze(0)
            pred_logits = class_queries_logits[i].unsqueeze(0).transpose(1, 2)
            loss_ce = criterion(pred_logits, target_classes)
            all_ce_loss.append(loss_ce)
        losses = {"loss_cross_entropy": torch.cat(all_ce_loss, dim=-1).mean()}
        return losses

    def loss_masks(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: list[torch.Tensor],
        indices: tuple[np.array],
        num_masks: int,
    ) -> dict[str, torch.Tensor]:
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # shape (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits[src_idx]
        # shape (batch_size, num_queries, height, width)
        # pad all and stack the targets to the num_labels dimension
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates
        pred_masks = pred_masks[:, None].to(torch.float32)
        target_masks = target_masks[:, None].to(torch.float32)

        # Sample point coordinates
        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)
        
        point_logits = sample_point(pred_masks, point_coordinates.to(pred_masks.dtype), align_corners=False).squeeze(1)

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    def _get_predictions_permutation_indices(self, indices):
        # Permute predictions following indices
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        # Permute labels following indices
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        In Mask2Former paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'
        for the foreground class in `classes`.

        Args:
            logits (`torch.Tensor`):
            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is:
            the number of foreground classes. The values are logits.

        Returns:
            scores (`torch.Tensor`): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most
            uncertain locations having the highest uncertainty score.
        """
        uncertainty_scores = -(torch.abs(logits))
        return uncertainty_scores

    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> torch.Tensor:
        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random point coordinates
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        # Get sampled prediction value for the point coordinates
        point_logits = sample_point(logits, point_coordinates.to(logits.dtype), align_corners=False)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1,
            )
        return point_coordinates

    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: list[torch.Tensor],
        class_labels: list[torch.Tensor],
        auxiliary_predictions: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, num_labels)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            class_labels (`list[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            auxiliary_predictions (`dict[str, torch.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], then it contains the logits from
                the inner layers of the Mask2FormerMaskedAttentionDecoder.

        Returns:
            losses (`dict[str, Tensor]`): A dict of `torch.Tensor` containing three keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid cross_entropy loss on the predicted and ground truth
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], the dictionary contains additional
            losses for each auxiliary predictions.
        """

        # retrieve the matching between the outputs of the last layer and the labels
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        # compute the average number of target masks for normalization purposes
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        # get all the losses
        losses: dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["pred_masks"]
                class_queries_logits = aux_outputs["pred_logits"]
                loss_dict = self.forward(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses

    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        num_masks = sum([len(classes) for classes in class_labels])
        num_masks = torch.as_tensor(num_masks, dtype=torch.float, device=device)
        world_size = 1
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                num_masks = reduce(num_masks)
                world_size = PartialState().num_processes

        num_masks = torch.clamp(num_masks / world_size, min=1)
        return num_masks


    

# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
class Mask2FormerHungarianMatcher(nn.Module):
    """This class computes an assignment between the labels and the predictions of the network.

    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0, num_points: int = 12544
    ):
        """Creates the matcher

        Params:
            cost_class (`float`, *optional*, defaults to 1.0):
                Relative weight of the classification error in the matching cost.
            cost_mask (`float`, *optional*,  defaults to 1.0):
                This is the relative weight of the focal loss of the binary mask in the matching cost.
            cost_dice (`float`, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost.
            num_points (`int`, *optional*, defaults to 12544):
                No. of points to sample on which the mask loss will be calculated. The same set of K points are
                uniformly sampled for all prediction and ground truth masks to construct the cost matrix for bipartite
                matching.
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs can't be 0")

        self.num_points = num_points
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> list[tuple[Tensor]]:
        indices: List[Tuple[np.array]] = []

        # iterate through batch size
        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class]. The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)
            target_mask = target_mask[:, None].to(torch.float32)
            pred_mask = pred_mask[:, None].to(torch.float32)

            # Sample ground truth and predicted masks
            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)

            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1).to(target_mask.dtype)
            target_mask = sample_point(target_mask, target_coordinates, align_corners=False).squeeze(1)

            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1).to(pred_mask.dtype)
            pred_mask = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)

            # compute the cross entropy loss between each mask pairs -> shape (num_queries, num_labels)
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            # Compute the dice loss betwen each mask pairs -> shape (num_queries, num_labels)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            # final cost matrix
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            # eliminate infinite values in cost_matrix to avoid the error ``ValueError: cost matrix is infeasible``
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            cost_matrix = torch.nan_to_num(cost_matrix, 0)
            # do the assigmented using the hungarian algorithm in scipy
            assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix.to(torch.float32).cpu())
            indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
        ]
        return matched_indices
    
# Adapted from https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py
def sample_point(
    input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=False, **kwargs
) -> torch.Tensor:
    """
    A wrapper around `torch.nn.functional.grid_sample` to support 3D point_coordinates tensors.

    Args:
        input_features (`torch.Tensor` of shape (batch_size, channels, height, width)):
            A tensor that contains features map on a height * width grid
        point_coordinates (`torch.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,:
        2)):
            A tensor that contains [0, 1] * [0, 1] normalized point coordinates
        add_dim (`bool`):
            boolean value to keep track of added dimension

    Returns:
        point_features (`torch.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,
        height_grid, width_grid):
            A tensor that contains features for points in `point_coordinates`.
    """
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    # use nn.function.grid_sample to get features for points in `point_coordinates` via bilinear interpolation
    point_features = torch.nn.functional.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        point_features = point_features.squeeze(3)

    return point_features

# Copied from transformers.models.maskformer.modeling_maskformer.dice_loss
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follows:

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss


def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, num_masks: int) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss = criterion(inputs, labels)

    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss

# Copied from transformers.models.maskformer.modeling_maskformer.pair_wise_dice_loss
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `torch.Tensor`: The computed loss between each pairs.
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    # using broadcasting to get a [num_queries, NUM_CLASSES] matrix
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""
    A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss between each pairs.
    """

    height_and_width = inputs.shape[1]

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    loss_pos = torch.matmul(cross_entropy_loss_pos / height_and_width, labels.T)
    loss_neg = torch.matmul(cross_entropy_loss_neg / height_and_width, (1 - labels).T)
    loss = loss_pos + loss_neg
    return loss
