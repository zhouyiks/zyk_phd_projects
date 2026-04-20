import argparse
import copy
import os.path as osp
import torch
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
import os

def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth-model', help='pth model file')
    parser.add_argument(
        '--save-path', type=str, default='./work_dirs/hf_model', help='save folder name')
    args = parser.parse_args()
    return args

@master_only
def master_print(msg):
    print(msg)

def main():
    args = parse_args()

    # build model
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)
    backend = get_file_backend(args.pth_model)

    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    model._merge_lora()
    model.mllm.transfer_to_hf = True

    all_state_dict = model.all_state_dict()
    name_map = {'mllm.model.': 'mllm.', '.gamma': '.g_weight'}

    all_state_dict_new = {}
    for key in all_state_dict.keys():
        new_key = copy.deepcopy(key)
        for _text in name_map.keys():
            new_key = new_key.replace(_text, name_map[_text])
        all_state_dict_new[new_key] = all_state_dict[key]

    # build the hf format model
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from projects.llava_sam2.hf.dense360_models.configuration_dense360_chat import Dense360ChatConfig
    from projects.llava_sam2.hf.dense360_models.modeling_dense360_chat import Dense360ChatModel
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (Qwen2_5_VLPreTrainedModel, Qwen2_5_VLForConditionalGeneration)

    qwen2_5vl_3b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto")

    mllm_config = qwen2_5vl_3b.config
    # mllm_config = Qwen2_5_VLConfig.from_pretrained(cfg.path, trust_remote_code=True)
    config_dict = {}
    config_dict['mllm_config'] = mllm_config.to_dict()
    config_dict['auto_map'] = \
        {
            'AutoConfig': 'configuration_dense360_chat.Dense360ChatConfig',
            'AutoModel': 'modeling_dense360_chat.Dense360ChatModel',
        }
    
    # if len(model.tokenizer) > config_dict['mllm_config']["vocab_size"]:
    config_dict['mllm_config']["vocab_size"] = len(model.tokenizer)
    # else:
    #     all_state_dict_new_new = {}
    #     for key in all_state_dict_new.keys():
    #         new_key = copy.deepcopy(key)
    #         if new_key in ['mllm.lm_head.weight', 'mllm.model.embed_tokens.weight']:
    #             old_lm_head = all_state_dict_new[new_key]
    #             padded_lm_head = torch.zeros((config_dict['mllm_config']["vocab_size"], old_lm_head.shape[1]), dtype=old_lm_head.dtype)
    #             all_state_dict_new_new[new_key] = padded_lm_head
    #         else:
    #             all_state_dict_new_new[new_key] = all_state_dict_new[new_key]
    
    # config_dict['mllm_config']["model_type"] = "dense360_chat"
    dense360_hf_config = Dense360ChatConfig(**config_dict)
    hf_dense360_model = Dense360ChatModel(dense360_hf_config, mllm_model=model.mllm.model)

    hf_dense360_model.load_state_dict(all_state_dict_new)
    hf_dense360_model.mllm.lm_head.weight = copy.deepcopy(model.mllm.model.lm_head.weight)
    hf_dense360_model.mllm.visual = copy.deepcopy(qwen2_5vl_3b.visual)

    hf_dense360_model.save_pretrained(args.save_path)
    model.tokenizer.save_pretrained(args.save_path)
    model.mllm.processor.save_pretrained(args.save_path)
    print(f"Save the hf model into {args.save_path}")

    # copy the files
    os.system(f"cp -pr ./projects/llava_sam2/hf/dense360_models/* {args.save_path}")

if __name__ == '__main__':
    main()