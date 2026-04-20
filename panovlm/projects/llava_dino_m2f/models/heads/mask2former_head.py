# Modified from https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/segmentation/models/heads/mask2former_head.py
# 
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, Tuple

from torch import nn
from torch.nn import functional as F

from projects.llava_dino_m2f.models.heads.pixel_decoder import MSDeformAttnPixelDecoder
from projects.llava_dino_m2f.models.heads.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


class Mask2FormerHead(nn.Module):
    def __init__(
        self,
        input_shape: Dict[str, Tuple[int]],  # ShapeSpec: [channels, height, width, stride]
        hidden_dim: int = 2048,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_in_feature: str = "multi_scale_pixel_decoder",
        num_proposal_layers: int = 3,
        num_queries: int = 100,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        orig_input_shape = input_shape
        input_shape = sorted(input_shape.items(), key=lambda x: x[1][-1])
        self.in_features = [k for k, _ in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=orig_input_shape,
            transformer_dropout=0.0,
            transformer_nheads=16,
            transformer_dim_feedforward=4096,
            transformer_enc_layers=6,
            conv_dim=hidden_dim,
            mask_dim=hidden_dim,
            norm="GN",
            transformer_in_features=["1", "2", "3", "4"],
            common_stride=4,
        )
        self.predictor = MultiScaleMaskedTransformerDecoder(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=16,
            dim_feedforward=4096,
            dec_layers=9,
            pre_norm=False,
            mask_dim=hidden_dim,
            enforce_input_project=False,
            num_proposal_layers=num_proposal_layers,
        )

        self.transformer_in_feature = transformer_in_feature

    def forward_features(self, features, mask=None):
        return self.layers(features, mask)

    def forward(self, features, mask=None):
        output = self.forward_features(features, mask)
        return output

    def predict(self, features, mask=None, rescale_to=(512, 512)):
        output = self.forward_features(features, mask)
        output["pred_masks"] = F.interpolate(
            output["pred_masks"],
            size=rescale_to,
            mode="bilinear",
            align_corners=False,
        )
        return output

    def layers(self, features, mask=None):
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        return predictions