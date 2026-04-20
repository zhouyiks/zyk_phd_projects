from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR, OptimWrapper
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from peft import LoraConfig

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss

from projects.llava_dino_m2f.datasets import (
    RSDenseSegDataset, llava_dino_m2f_collate_fn)
from projects.llava_dino_m2f.models.llava_dino_m2f import LLaVADINOM2F


#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = './OpenGVLab/InternVL3-2B'
pretrained_pth = None
num_m2f_queries = 200
dinov3_path = "./pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
dinov3_input_size = 224

work_dir = "work_dirs/internvl3_dino_m2f_2b"

# Data
template = "internlm2_chat"
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
dataloader_num_workers = 1
max_epochs = 1
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
        trust_remote_code=True,
    ),
    llm_lora=dict(
        type=LoraConfig,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    ),
    freeze_llm=True,
    tokenizer=tokenizer,
    backbone_name="dinov3_vitl16",
    backbone_weights=dinov3_path,
    num_queries=num_m2f_queries,
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
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

bhp_watertanks_pool_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/BHP_Watertanks_Pool.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

bhp_watertanks_watertank_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/BHP_Watertanks_Watertank.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

bsb_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/BSB.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

casid_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/CASID.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

chn6cug_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/CHN6-CUG.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

cityosm_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/CITY-OSM.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

crowdai_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/CrowdAI.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

dgroad_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/DeepGlobe_Road_CVPR2018.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

dglandcover_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/DGLandCover.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

dlrsd_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/DLRSD.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

evlabss_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/EVLab_SS.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

fast_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/FAST.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

finegrip_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/FineGrip.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

flair_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/FLAIR.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

globe230k_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/Globe230k.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

inria_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/inria.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

landcoverai_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/LandCoverAI.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

loveda_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/LoveDA.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

lrsny_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/LRSNY.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

njds_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/NJDS.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

nwpu_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/NWPU.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

oem_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/OEM.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

ottawa_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/Ottawa.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

postdam_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/Postdam.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

sior_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/SIOR.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

sota_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/SOTA.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

udd5_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/UDD_5.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

udd6_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/UDD_6.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

whugid_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/WHU_GID.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

whumix_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/WHU_MIX_512.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

summer_dataset = dict(
    type=RSDenseSegDataset,
    image_folder='/media/disk3/dataset/zipped_formatted_datasets/',
    data_path='/media/disk3/dataset/zipped_formatted_datasets/Zurich_Summer.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    random_ratio=0.5,
    dinov3_input_size=dinov3_input_size,
    dinov3_input_mean=(0.430, 0.411, 0.296),
    dinov3_input_std=(0.213, 0.156, 0.143),
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        bhp_watertanks_pool_dataset,
        bhp_watertanks_watertank_dataset,
        bsb_dataset,
        casid_dataset,
        chn6cug_dataset,
        cityosm_dataset,
        # crowdai_dataset,
        dgroad_dataset,
        dglandcover_dataset,
        dlrsd_dataset,
        evlabss_dataset,
        # fast_dataset,
        # finegrip_dataset,
        flair_dataset,
        globe230k_dataset,
        inria_dataset,
        landcoverai_dataset,
        loveda_dataset,
        lrsny_dataset,
        # njds_dataset,
        nwpu_dataset,
        oem_dataset,
        ottawa_dataset,
        postdam_dataset,
        # sior_dataset,
        # sota_dataset,
        udd5_dataset,
        udd6_dataset,
        whugid_dataset,
        whumix_dataset,
        summer_dataset,
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
