from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss


from projects.llava_dino_m2f.datasets import SAMRSDataset, llava_dino_m2f_collate_fn
from projects.llava_dino_m2f.models.llava_dino_m2f import LLaVADINOM2F


#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = './OpenGVLab/InternVL2_5-4B'
pretrained_pth = None
num_m2f_queries = 200
dinov3_path = "./pretrained_weights/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

# Data
template = "internlm2_chat"
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 4
dataloader_num_workers = 2
max_epochs = 5
optim_type = AdamW
# official 1024 -> 4e-5
# lr = 1e-6
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 1000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# version 0
PROPOSAL_TOKENS = ['[SEG{id}]'.format(id=str(i).zfill(3)) for i in range(num_m2f_queries)]
special_tokens = ['<p>', '</p>', '[CLS]', '[BG_CLS]', '<obj>', '</obj>', '<OBJ_CONTEXT>'] + PROPOSAL_TOKENS

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################

model = dict(
    type=LLaVADINOM2F,
    special_tokens=special_tokens,
    mllm=dict(
        type=AutoModel.from_pretrained,
        pretrained_model_name_or_path=path,
    ),
    tokenizer=tokenizer,
    backbone_name="dinov3_vitl16",
    backbone_weights=dinov3_path,
    num_m2f_queries=num_m2f_queries,
    num_proposal_layers=3,
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5),
    loss_sample_points=True,
    pretrained_pth=pretrained_pth,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

samrs_fast_insseg_dataset = dict(
    type=SAMRSDataset,
    image_folder='./data/samrs/fast/images',
    data_path='./data/samrs/fast/ins',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=1024,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

samrs_sior_insseg_dataset = dict(
    type=SAMRSDataset,
    image_folder='./data/samrs/sior/JPEGImages-trainval',
    data_path='./data/samrs/sior/ins',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=1024,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

samrs_sota_insseg_dataset = dict(
    type=SAMRSDataset,
    image_folder='./data/samrs/sota/images',
    data_path='./data/samrs/sota/ins',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=1024,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

samrs_sotarbb_insseg_dataset = dict(
    type=SAMRSDataset,
    image_folder='./data/samrs/sota_rbb/images',
    data_path='./data/samrs/sota_rbb/ins',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=1024,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)


train_dataset = dict(
    type=ConcatDataset, datasets=[
        samrs_fast_insseg_dataset,
        samrs_sior_insseg_dataset,
        samrs_sota_insseg_dataset,
        samrs_sotarbb_insseg_dataset,
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
    collate_fn=dict(type=llava_dino_m2f_collate_fn)
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
    dtype='bfloat16'
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
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

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
