from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop

from projects.samtok.models.sam2 import SAM2Config, VQ_SAM2Config, VQ_SAM2
from projects.samtok.models.vq_sam2 import VQ_SAM2Model
from projects.samtok.models import DirectResize
from projects.samtok.datasets.entity_dataset import EntityDataset
from projects.samtok.datasets.sa1b_dataset import SA1BDataset
from projects.samtok.datasets.collect_fns import vq_sam2_collate_fn

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
sam2_path = "pretrained_weights/sam2.1_hiera_large.pt"
pretrained_pth = None

work_dir = "work_dirs/rs_vq_sam2_8192x1_16tokens_true_causal_attn"

# Scheduler & Optimizer
batch_size = 16  # per_device
accumulative_counts = 1
dataloader_num_workers = 16
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 5000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)


#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
sam2_config = dict(
    type=SAM2Config,
    cfg_path="sam2.1_hiera_l.yaml",
    ckpt_path=sam2_path,
    num_mask_tokens=16,
    is_causal=True,
)

vq_sam2_config = dict(
    type=VQ_SAM2Config,
    sam2_config=sam2_config,
    codebook_size=8192,
    codebook_depth=1,
    shared_codebook=False,
    latent_dim=256,
    loss_sample_points=True,
    vq_loss_weight=1.0,
)

model = dict(
    type=VQ_SAM2Model,
    hf_model=dict(
        type=VQ_SAM2,
        config=vq_sam2_config,
    ),
    sam2_pretrained_weights=sam2_path,
    pretrained_pth=pretrained_pth,
    freeze_sam2_decoder=False,
    box_input=False,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

sam2_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)

DATA_ROOT = ''
rs_segdata_info_json = "./data/rs_mask_info_20260116.json"
sam_info_json = "./data/sam_info.json"

sft_seg_dataset_part2 = dict(
    type=EntityDataset,
    image_folder=DATA_ROOT,
    preprocessor=sam2_image_processor,
    repeats=10,
    entity_info_json=rs_segdata_info_json,
)

sft_seg_dataset_earthreason = dict(
    type=EntityDataset,
    image_folder=DATA_ROOT,
    preprocessor=sam2_image_processor,
    repeats=10,
    entity_info_json="./data/earthreason_mask_info_20260228.json",
)

sft_seg_dataset_lasers = dict(
    type=EntityDataset,
    image_folder=DATA_ROOT,
    preprocessor=sam2_image_processor,
    repeats=10,
    entity_info_json="./data/lasers_mask_info_20260227.json",
)

sft_seg_dataset_refsegrs = dict(
    type=EntityDataset,
    image_folder=DATA_ROOT,
    preprocessor=sam2_image_processor,
    repeats=10,
    entity_info_json="./data/refsegrs_mask_info_20260227.json",
)

sft_seg_dataset_risbench = dict(
    type=EntityDataset,
    image_folder=DATA_ROOT,
    preprocessor=sam2_image_processor,
    repeats=10,
    entity_info_json="./data/risbench_mask_info_20260228.json",
)

sft_seg_dataset_rrsis_d = dict(
    type=EntityDataset,
    image_folder=DATA_ROOT,
    preprocessor=sam2_image_processor,
    repeats=10,
    entity_info_json="./data/RRSIS_D_mask_info_20260227.json",
)

sa1b_dataset = dict(
    type=SA1BDataset,
    image_folder=DATA_ROOT,
    preprocessor=sam2_image_processor,
    multi_targets=True,
    repeats=10,
    fast_load=True,
    sam_info_json=sam_info_json,
    scan_record_folder='./left_sa1b_indices/vq_sam2_4096x1_8tokens_false_causal_attn/'
)



train_dataset = dict(
    type=ConcatDataset, datasets=[
        sft_seg_dataset_part2, sa1b_dataset, 
        sft_seg_dataset_earthreason,
        sft_seg_dataset_lasers,
        sft_seg_dataset_refsegrs,
        sft_seg_dataset_risbench,
        sft_seg_dataset_rrsis_d,
    ]
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=vq_sam2_collate_fn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float32'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='hccl'),
)

# set visualizer
# visualizer = None
from mmengine.visualization import Visualizer, TensorboardVisBackend
visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)