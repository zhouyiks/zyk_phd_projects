from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from third_parts.mmdet.models.losses import CrossEntropyLoss

from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict

from .lisa import LisaModel

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria
from transformers import GenerationConfig
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

import numpy as np

from .internvl import InternVL_Slowfast
from .utils import dynamic_preprocess

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from pycocotools import mask as _mask

from types import MethodType

from xtuner.model.utils import guess_load_checkpoint

from mmcv.ops import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness

from transformers import Mask2FormerForUniversalSegmentation, AutoModel

# from dam import DescribeAnythingModel, disable_torch_init

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

from ..datasets.utils.constants import (SEG_TOKEN, CLS_TOKEN, BG_CLS_TOKEN, PHRASE_START_TOKEN, PHRASE_END_TOKEN)


class VideoLLaVASAMM2FModel(LisaModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 special_tokens=None,
                 vocab_embeds_name="tok_embeddings",
                 lm_head_name="output",
                 # mask loss
                 loss_sample_points=False,
                 num_points=12544,
                 # for arch selection
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 template=None,
                 # for inference large model
                 split_model=False,
                 # ext
                 preprocessor=None,
                 # bs
                 bs:int=0,
                 # mask2former
                 mask2former=None,
                 num_m2f_queries=300,
                 num_m2f_proposals=100,
                 # sam2
                 grounding_encoder=None,
                 frozen_sam2_decoder=True,
                 ):
        super(LisaModel, self).__init__()

        self.PROPOSAL_TOKENS = [SEG_TOKEN.format(id=str(i).zfill(3)) for i in range(num_m2f_proposals)]
        self.split_model = split_model
        if split_model:
            mllm.model_split = split_model
        if special_tokens is None:
            special_tokens = [CLS_TOKEN, BG_CLS_TOKEN].extend(self.PROPOSAL_TOKENS)
            # special_tokens = [CLS_TOKEN, BG_CLS_TOKEN, SEG_TOKEN.format(id="")]
        self.special_tokens = special_tokens
        if 'special_tokens' not in mllm.keys():
            mllm.special_tokens = special_tokens
        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type
        self.vocab_embeds_name=vocab_embeds_name
        self.lm_head_name=lm_head_name

        self.tokenizer = BUILDER.build(tokenizer)
        
        self._add_special_tokens()

        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)
            elif self.arch_type == 'qwen':
                self.mllm.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.get_output_embeddings().weight.requires_grad_(True)
            elif self.arch_type == 'llava':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)

        if self.arch_type == 'intern_vl':
            in_dim = self.mllm.model.config.llm_config.hidden_size
        elif self.arch_type == 'qwen':
            in_dim = self.mllm.model.config.hidden_size
        elif self.arch_type == 'llava':
            # for llava, the hidden size is in language model
            in_dim = self.mllm.model.language_model.config.hidden_size
        
        if preprocessor is None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = BUILDER.build(preprocessor)
        
        '''
        self.mask2former = BUILDER.build(mask2former)
        assert self.mask2former.config.num_queries == num_m2f_queries
        self.num_m2f_queries = num_m2f_queries
        self.num_m2f_proposals = num_m2f_proposals
          
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

        out_dim = self.mask2former.config.hidden_dim

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
        '''
        
        # sam2
        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)

        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.template = template
        self.bs = bs


    def _merge_lora(self):
        # print('pre merge lora: ', self.mllm.model.language_model.base_model.model.get_input_embeddings().weight.shape)
        try:
            self.mllm.model.model = self.mllm.model.model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.visual = self.mllm.model.visual.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        # print('after merge lora: ', self.mllm.model.language_model.get_input_embeddings().weight.shape)
        return

    def all_state_dict(self, *args, **kwargs):
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        return state_dict

    def activation_checkpointing_disable(self):
        if self.arch_type == 'qwen':
            self.mllm.model.model.gradient_checkpointing_disable()
        else:
            self.mllm.model.language_model.gradient_checkpointing_disable()


    def _add_special_tokens(self):
        special_tokens = self.special_tokens
        _num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        # self.the_first_seg_token_idx = self.tokenizer(self.PROPOSAL_TOKENS[0], add_special_tokens=False).input_ids[0]
        # self.the_last_seg_token_idx = self.tokenizer(self.PROPOSAL_TOKENS[-1], add_special_tokens=False).input_ids[0]
        self.cls_token_idx = self.tokenizer(CLS_TOKEN, add_special_tokens=False).input_ids[0]
        # self.bg_cls_token_idx = self.tokenizer(BG_CLS_TOKEN, add_special_tokens=False).input_ids[0]

    def state_dict(self, *args, **kwargs):
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        from collections import OrderedDict

        to_return = OrderedDict()
        # Step 1. visual_encoder. By convention, we do not fine-tune the visual encoder.
        if self.mllm.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.mllm.model.vision_model, state_dict=state_dict))
            raise NotImplementedError
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
            raise NotImplementedError
        # Step 2. LLM
        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
            elif self.arch_type == 'qwen':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.model, state_dict=state_dict)
                )
            elif self.arch_type == 'llava':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
        elif not self.mllm.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
            raise NotImplementedError
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
                for k, v in state_dict.items() if 'mask2former.' in k
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
            {
                k: v
                for k, v in state_dict.items() if 'mask2former.' in k
            }
        )
        to_return.update(
            {k: v
             for k, v in state_dict.items() if f'{self.lm_head_name}.weight' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if f'{self.vocab_embeds_name}.weight' in k})
        
        # SAM/SAM2
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mask_decoder' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_hidden_fcs.' in k})

        return to_return
    
    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=20):
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            # assert len(pred_mebeds) == len(gt_masks)
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video
    
    def _get_pesudo_data(self, dtype, device):
        assert self.bs > 0
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks
    
    def forward(self, data, data_samples=None, mode='loss'):
        # for n, p in self.named_parameters():
        #     if p.requires_grad:
        #         print(n)
        # exit(0)
        
        # 0. encode multi-scale visual features into 100~300 queries
        if 'm2f_inputs' in data:
            m2f_inputs = data.pop('m2f_inputs')
            m2f_inputs['pixel_values'] = m2f_inputs['pixel_values'].to(self.mask2former.dtype)
            m2f_inputs['pixel_mask'] = m2f_inputs['pixel_mask'].to(self.mask2former.dtype)
            query_features, pixel_level_module_output = \
                self.mask2former.forward_first_part(**m2f_inputs)
            query_embeds = self.m2f_to_llm(query_features)  # BS, m2f_NQ, 2048
            data['query_embeds'] = query_embeds

        output = self.mllm(data, data_samples, mode)
        
        input_ids = data.get('input_ids', None)
        g_pixel_values = data.get('g_pixel_values', None)
        gt_masks = data.get('masks', None)

        frames_per_batch = data.get('frames_per_batch', None)

        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]
        
        # 1. refseg
        refseg_token_mask = input_ids == self.cls_token_idx
        # panseg_token_mask = (input_ids >= self.the_first_seg_token_idx) & (input_ids <= self.the_last_seg_token_idx)
        # refseg_token_mask[panseg_token_mask.sum(1)>0, :] = False

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])
        
        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[refseg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero
        
        refseg_token_counts = refseg_token_mask.int().sum(-1)
        if not seg_valid:
            refseg_token_counts += 5
        
        pred_embeddings_list_ = torch.split(pred_embeddings, refseg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        if not success:
            # raise NotImplementedError
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )

            ori_size_list = []
            for i_bs, mask in enumerate(gt_masks):
                mask_shape = mask.shape[-2:]
                ori_size_list += [mask_shape] * frames_per_batch[i_bs]
            
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero
            refseg_token_counts = refseg_token_counts - refseg_token_counts + 5

            pred_embeddings_list_ = torch.split(pred_embeddings, refseg_token_counts.tolist(), dim=0)
            pred_embeddings_list = []
            for item in pred_embeddings_list_:
                if len(item) != 0:
                    pred_embeddings_list.append(item)
            pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
                pred_embeddings_list, frames_per_batch)
            assert success


        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )

        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)

        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
        }

        return loss_dict

        
        # # 2. panoseg
        # panseg_token_mask = (input_ids >= self.the_first_seg_token_idx) & (input_ids <= self.the_last_seg_token_idx)
        # refseg_token_counts = refseg_token_mask.int().sum(-1)
        # if panseg_token_mask.sum() == 0:
        #     seg_valid = False
        # else:
        #     seg_valid = True
        #     # 1.1 extract proposals hidden
        #     gt_masks = data.pop('masks')
        #     gt_class_ids = data.pop('class_ids')


        # if gt_masks is None:
        #     seg_valid = False
        # else:
        #     seg_valid = True

        # input_ids = data['input_ids']
        # seg_token_mask = (input_ids >= self.the_first_seg_token_idx) & (input_ids <= self.the_last_seg_token_idx)
        # hidden_states = output.hidden_states
        # hidden_states = self.llm_to_m2f(hidden_states[-1])

        # _zero = hidden_states.mean() * 0.0
        # if seg_valid:
        #     pred_embeddings = hidden_states[seg_token_mask] + _zero
        # else:
        #     pred_embeddings = hidden_states[:, :self.num_m2f_proposals].flatten(0, 1) + _zero

        # seg_token_counts = seg_token_mask.int().sum(-1)
        # if not seg_valid:
        #     seg_token_counts += 5

        # pred_embeddings = pred_embeddings.reshape(-1, self.num_m2f_proposals, self.mask2former.config.hidden_dim * 2)
        # pred_embeddings = pred_embeddings.transpose(0, 1)

        # # 1.2 extract [CLS] tokens
        # bs = input_ids.shape[0]
        # bg_cls_token_id = torch.as_tensor([self.bg_cls_token_idx,], dtype=input_ids.dtype, device=input_ids.device)
        # bg_cls_embedding = self.mllm.model.language_model.get_input_embeddings()(bg_cls_token_id).clone()
        # bg_cls_embedding = bg_cls_embedding.unsqueeze(0).repeat(bs, 1, 1)
        # cls_token_mask = input_ids == self.cls_token_idx
        # hidden_states = output.hidden_states
        # hidden_states = self.llm_to_cls(torch.cat([hidden_states[-1], bg_cls_embedding], dim=1))
        # bg_cls_embedding = hidden_states[:, -1:, :]
        # hidden_states = hidden_states[:, :-1, :]
        # _zero = hidden_states.mean() * 0.0
        # if seg_valid:
        #     text_classifier = hidden_states[cls_token_mask] + _zero
        # else:
        #     text_classifier = hidden_states[:, :1].flatten(0, 1) + _zero
        
        # cls_token_counts = cls_token_mask.int().sum(-1)
        # if not seg_valid:
        #     cls_token_counts += 1
        
        # text_classifier_list_ = torch.split(text_classifier, cls_token_counts.tolist(), dim=0)
        # text_classifier_list = []
        # for bs_i, item in enumerate(text_classifier_list_):
        #     if len(item) != 0:
        #         text_classifier_list.append(torch.cat([item, bg_cls_embedding[bs_i]], dim=0))
  
        # # 2. proposals go through mask2former decoder layers
        # m2f_outputs = self.mask2former.forward_second_part(
        #     query_features=pred_embeddings[:, :, :self.mask2former.config.hidden_dim], # q, b, c
        #     query_embeddings=pred_embeddings[:, :, self.mask2former.config.hidden_dim:], # q, b, c
        #     pixel_level_module_output=pixel_level_module_output,
        #     text_classifier=text_classifier_list,
        #     mask_labels=gt_masks,
        #     class_labels=gt_class_ids,
        #     **m2f_inputs
        # )

        # m2f_loss = m2f_outputs.loss
        # if not seg_valid:
        #     _scale = 0.0
        # else:
        #     _scale = 1.0
        # m2f_loss = m2f_loss * _scale + self.mask2former.class_predictor.weight.sum() * 0.0

        # loss_dict = {
        #     'llm_loss': output.loss,
        #     'm2f_loss': m2f_loss,
        # }
        # return loss_dict

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)
    
    def genetate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        if len(pred_embeddings_list) == len(frames_per_batch):
            success = True
        else:
            success = False
            print("len(pred_embeddings_list):{} is not equal to len(frames_per_batch):{} !!!".format(len(pred_embeddings_list), len(frames_per_batch)))
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video, success

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation(
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


class VideoLLaVASAMM2FModel_zero3(VideoLLaVASAMM2FModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=['[SEG]', ],
                 loss_sample_points=False,
                 num_points=12544,
                 # for slow fast arch
                 fast_pool=False,
                 fast_pool_size=4,
                 arch_type='intern_vl',
                 # zero3
                 bs=1,
                 ):
        super(VideoLLaVASAMM2FModel_zero3, self).__init__(
            mllm=mllm,
            tokenizer=tokenizer,
            grounding_encoder=grounding_encoder,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            torch_dtype=torch_dtype,
            pretrained_pth=pretrained_pth,
            frozen_sam2_decoder=frozen_sam2_decoder,
            special_tokens=special_tokens,
            loss_sample_points=loss_sample_points,
            num_points=num_points,
            # for slow fast arch
            fast_pool=fast_pool,
            fast_pool_size=fast_pool_size,
            arch_type=arch_type,
        )
        self.bs = bs

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)

        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        # print('frmaes_per_batch: ', frames_per_batch)
        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        if not success:
            raise NotImplementedError
            # return {'llm_loss': output.loss, 'loss_mask': output.loss * 0.0, 'loss_dice': output.loss * 0.0}

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        # print(f"Done, {g_pixel_values.device} !!!\n\n")
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        # print(f"Done, {g_pixel_values.device} !!! {num_frames}---{num_objs}, {language_embeddings.shape}\n\n")
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)
        # pred_masks = torch.cat(pred_masks, dim=0)


        bs = len(pred_masks)
        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
        }
        return loss_dict
