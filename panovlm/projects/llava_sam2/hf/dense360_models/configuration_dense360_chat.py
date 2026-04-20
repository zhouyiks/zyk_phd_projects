# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Dense360ChatConfig(PretrainedConfig):
    model_type = 'dense360_chat'
    is_composition = True

    def __init__(
            self,
            mllm_config=None,
            **kwargs):
        super().__init__(**kwargs)
        if mllm_config is None:
            mllm_config = {"architectures": ["Qwen2_5_VLForConditionalGeneration"]}
            logger.info('mllm_config is None. Initializing the Qwen2_5_VLForConditionalGeneration  with default values.')
        
        self.mllm_config = Qwen2_5_VLConfig(**mllm_config)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['mllm_config'] = self.mllm_config.to_dict()

        return output
