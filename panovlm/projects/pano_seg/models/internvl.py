import os
import torch
from xtuner.model import InternVL_V1_5
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
import torch.nn as nn

from mmengine import print_log
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BitsAndBytesConfig)
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from xtuner.model.utils import (find_all_linear_names, get_peft_model_state_dict,
                    guess_load_checkpoint, make_inputs_require_grad)
from ..datasets.utils.constants import IMG_CONTEXT_TOKEN, OBJ_CONTEXT_TOKEN, IMG_START_TOKEN
from .mask2former import M2FRegionEncoder


def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

# This function is used to split large model
def split_model(model_name):
    import math
    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.2))
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.8)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = rank
    device_map['language_model.model.norm'] = rank
    device_map['language_model.lm_head'] = rank
    device_map[f'language_model.model.layers.{num_layers - 1}'] = rank
    return device_map

class InternVL_Slowfast(InternVL_V1_5):

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
                 m2f_path=None,
                 ):
        print_log('Start to load InternVL_V1_5 model.', logger='current')
        super(InternVL_V1_5, self).__init__()
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

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'

        if quantization_vit is False and quantization_llm is False:
            quantization = None
        else:
            llm_int8_skip_modules = ['mlp1']
            if quantization_llm and not quantization_vit:
                llm_int8_skip_modules.append('vision_model')

            if quantization_vit and not quantization_llm:
                llm_int8_skip_modules.append('language_model')

            quantization_config = dict(
                type=BitsAndBytesConfig,
                llm_int8_skip_modules=llm_int8_skip_modules,
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4')
            quantization_clazz = quantization_config.pop('type')
            quantization = quantization_clazz(**quantization_config)

        if model_split:
            # print("\n\nDone Model Split !!!!!!!!!!!\n\n")
            device_map = split_model("InternVL2-26B")
            # print(device_map)
            self.device = 'cuda'
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=device_map).eval()

        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization,
                config=config,
                trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.tokenizer = tokenizer

        if special_tokens is not None:
            self._add_special_tokens(special_tokens)

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.vision_model.requires_grad_(False)

        if hasattr(self.model.language_model, 'enable_input_require_grads'):
            self.model.language_model.enable_input_require_grads()
        else:
            self.model.language_model.get_input_embeddings(
            ).register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self._count = 0
        print_log(self, logger='current')
        print_log('InternVL_V1_5 construction is complete', logger='current')

        self.transfer_to_hf = False


        m2f_config = AutoConfig.from_pretrained(m2f_path)
        self.region_encoder = M2FRegionEncoder(m2f_config)
        in_dim = config.llm_config.hidden_size
        out_dim = m2f_config.hidden_dim
        self.llm_to_m2f = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim * 2)
        )
        self.m2f_dim = out_dim
        self.m2f_to_llm = nn.Sequential(
            nn.LayerNorm(out_dim,),
            nn.Linear(out_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim)
        )

    def _add_special_tokens(self, special_tokens):
        num_new_tokens = self.tokenizer.add_tokens(
            special_tokens, special_tokens=True)

        if num_new_tokens > 0:
            self.model.language_model.resize_token_embeddings(len(self.tokenizer))

        self.vp_start_token_id = self.tokenizer.convert_tokens_to_ids('<vp>')
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.obj_context_token_id = self.tokenizer.convert_tokens_to_ids(OBJ_CONTEXT_TOKEN)
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.model.vision_model.dtype) for image in pixel_values], dim=0)
            tile_image_indices = torch.cat([torch.LongTensor([idx]*len(image)) for idx, image in enumerate(pixel_values)])
        else:
            raise NotImplementedError()

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        image_flags = data['image_flags']
        query_embeds = data['query_embeds']

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
            query_embeds=query_embeds,
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
        query_embeds: Optional[torch.Tensor] = None,
        vp_overall_mask=None,
        prompt_masks=None,
        m2f_masks=None,
        multi_scale_features=None,
        tile_image_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids).clone()

        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?

        vit_embeds = vit_embeds[image_flags == 1]
        tile_image_indices = tile_image_indices.to(image_flags.device)[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        self._count += 1

        if vp_overall_mask is not None and prompt_masks is not None:
            image_vp_embeds = []

            vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
            prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

            vp_overall_mask = vp_overall_mask[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask] # (n_img, hw, c)
            
            vp_index = 0
            for i_img in range(len(vit_embeds)):
                image_vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                img_index = tile_image_indices[i_img]
                if vp_overall_mask[i_img]:
                    tile_vit_embeds = overall_tile_vit_embeds[vp_index].reshape(1, -1, C)
                    tile_vit_2_m2f_embeds = self.llm_to_m2f(tile_vit_embeds)
                    
                    prompt_masks_i = prompt_masks[vp_index]
                    n_obj = len(prompt_masks_i)
                    objects_prompt_masks = prompt_masks_i.reshape(n_obj, -1)

                    m2f_masks_i = m2f_masks[vp_index]
                    vp_h, vp_w = m2f_masks_i.shape[-2:]
                    padded_prompt_masks = torch.zeros(size=(tile_vit_embeds.shape[1], vp_h, vp_w)).to(prompt_masks_i.device).to(prompt_masks_i.dtype)
                    
                    for prompt_mask_i, m2f_mask_i in zip(prompt_masks_i, m2f_masks_i):
                        object_prompt_mask = prompt_mask_i.flatten()
                        padded_prompt_masks[object_prompt_mask] = m2f_mask_i.unsqeeze(0)
                    
                    tile_vit_2_m2f_embeds = tile_vit_2_m2f_embeds.transpose(0, 1)

                    multi_scale_features_i = []
                    for single_scale_features in multi_scale_features:
                        multi_scale_features_i.append(single_scale_features[img_index:img_index+1])
                    
                    tile_m2f_embeds = self.region_encoder(
                        query_features=tile_vit_2_m2f_embeds[:, :, :self.m2f_dim],
                        query_embeddings=tile_vit_2_m2f_embeds[:, :, self.m2f_dim:],
                        multi_scale_features=multi_scale_features_i,
                        prompt_masks=padded_prompt_masks.unsqueeze(0),
                    )
                    tile_m2f_embeds = tile_m2f_embeds.transpose(0, 1)
                    tile_m2f_2_vit_embeds = self.m2f_to_llm(tile_m2f_embeds)
                    tile_vit_embeds = tile_vit_embeds + tile_m2f_2_vit_embeds
                    tile_vit_embeds = tile_vit_embeds.repeat(n_obj, 1, 1)

                    image_vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                    vp_index += 1
            image_vp_embeds = torch.cat(image_vp_embeds, dim=0)
        else:
            image_vp_embeds = None

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        if image_vp_embeds is None:
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
                input_embeds[selected] = image_vp_embeds.reshape(-1, C)
            except Exception as e:
                image_vp_embeds = image_vp_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vp_embeds.shape={image_vp_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(image_vp_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(image_vp_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(image_vp_embeds) + 1
                    image_vp_embeds = torch.cat([image_vp_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = image_vp_embeds[:n_token]

        # object queries
        query_embeds = query_embeds.to(input_embeds.dtype)
        selected = (input_ids == self.obj_context_token_id)

        try:
            input_embeds[selected] = query_embeds.reshape(-1, C)
        except Exception as e:
            query_embeds = query_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape='
                    f'{input_embeds[selected].shape}, '
                    f'query_embeds.shape={query_embeds.shape}')
            n_token = selected.sum()
            if n_token > len(query_embeds):
                print(f"Wrong !!! {n_token} image tokens in text but only {len(query_embeds)} query embeds !!!")
                expand_ratio = n_token // len(query_embeds) + 1
                query_embeds = torch.cat([query_embeds] * expand_ratio, dim=0)

            input_embeds[selected] = query_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
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
                -1, self.model.language_model.config.vocab_size)
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
        image_flags: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fast_token_idx=None,
        fast_pixel_values=None,
        prompt_masks=None,
        vp_overall_mask=None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.model.device
        assert self.model.img_context_token_id is not None

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
                        [image.to(self.model.vision_model.dtype) for image in pixel_values], dim=0)

                vit_embeds = self.model.extract_feature(pixel_values.to(device))
           
            vit_embeds = vit_embeds[image_flags == 1]
            
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def state_dict(self, *args, **kwargs):
        if self.transfer_to_hf:
            state_dict = super(InternVL_V1_5, self).state_dict(*args, **kwargs)
            return state_dict
        else:
            return super().state_dict(*args, **kwargs)


