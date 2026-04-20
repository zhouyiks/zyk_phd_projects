from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoImageProcessor, Mask2FormerForUniversalSegmentation

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.llava_sam2.models.internvl import InternVL_Slowfast

from projects.llava_sam2.models import VideoLLaVASAMModel
from projects.llava_sam2.datasets import VideoReVOSDataset, VideoMeVISDataset, VideoRefYoutubeVOSDataset, video_lisa_collate_fn, VideoSAM2Dataset
from projects.llava_sam2.datasets import VideoChatUniViDataset
from projects.llava_sam2.datasets import RefCOCOgGCGDataset, OpenPsgGCGDataset, FlickrGCGDataset, GranDfGCGDataset, OspreyDataset, OspreyDescriptionDataset, OspreyShortDescriptionDataset
from projects.llava_sam2.datasets import LLaVADataset
from projects.llava_sam2.datasets import ReferSegmDataset
from projects.llava_sam2.models.preprocess.image_resize import DirectResize
from projects.llava_sam2.datasets import (
    CoCoPanoSegDataset,
    SemSegDataset,
    FlairSemSegDataset,
    CHN6CUGSemSegDataset,
    CITYOSMSemSegDataset,
    DGLCSemSegDataset,
    DGROADSemSegDataset,
    DLRSDSemSegDataset,
    EVLabSemSegDataset,
    LRSNYSemSegDataset,
    OEMSemSegDataset,
    OTTAWASemSegDataset,
    POTSDAMSemSegDataset,
    SUMMERSemSegDataset,
    UAVIDSemSegDataset,
    UDD5SemSegDataset,
    UDD6SemSegDataset,
    LoveDASemSegDataset,
    AI4CSemSegDataset,
    BSBInsSegDataset,
    NWPUInsSegDataset,
    FineGripPanoSegDataset,
    ISAIDInsSegDataset,
    WHUMIXInsSegDataset,
    GlobalScaleDataset,
    SAMRSDataset,
    IndonesiaDataset,
)


from projects.llava_psam2.datasets.utils.constants import (
    PHRASE_START_TOKEN, PHRASE_END_TOKEN, CLS_TOKEN, BG_CLS_TOKEN, SEG_TOKEN, OBJ_START_TOKEN, OBJ_END_TOKEN, OBJ_CONTEXT_TOKEN)

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = './OpenGVLab/InternVL3-2B'
pretrained_pth = None
num_m2f_queries = 200
num_m2f_proposals = 100
mask2former_path = "./facebook/mask2former-swin-large-coco-panoptic"

# Data
template = "internlm2_chat"
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 4
dataloader_num_workers = 2
max_epochs = 3
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
save_total_limit = 5  # Maximum checkpoints to keep (-1 means unlimited)

# version 0
PROPOSAL_TOKENS = ['[SEG{id}]'.format(id=str(i).zfill(3)) for i in range(num_m2f_proposals)]
special_tokens = ['<p>', '</p>', '[CLS]', '[BG_CLS]', '<obj>', '</obj>', '<OBJ_CONTEXT>'] + PROPOSAL_TOKENS

# version 1
# special_tokens = ['<p>', '</p>', 'SEG', '[CLS]', '[BG_CLS]', '<obj>', '</obj>', '<OBJ_CONTEXT>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=VideoLLaVASAMModel,
    special_tokens=special_tokens,
    vocab_embeds_name="embed_tokens",
    lm_head_name="lm_head",
    mllm=dict(
        type=InternVL_Slowfast,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'),
        special_tokens=special_tokens,
    ),
    mask2former=dict(
        type=Mask2FormerForUniversalSegmentation.from_pretrained,
        pretrained_model_name_or_path=mask2former_path,
    ),
    tokenizer=tokenizer,
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
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    # loss_sample_points=False,
    bs=batch_size,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

DATA_ROOT = './data/'
VIDEO_DATA_ROOT = DATA_ROOT + 'video_datas/'

############### video res
data_root_revos = VIDEO_DATA_ROOT + 'revos/'
video_revos_image_folder = data_root_revos
video_revos_expression_file = data_root_revos + 'meta_expressions_train_.json'
video_revos_mask_file = data_root_revos + 'mask_dict.json'

data_root_mevis = VIDEO_DATA_ROOT + 'mevis/train/'
video_mevis_image_folder = data_root_mevis + 'JPEGImages'
video_mevis_expression_file = data_root_mevis + 'meta_expressions.json'
video_mevis_mask_file = data_root_mevis + 'mask_dict.json'

data_root_refytvos = VIDEO_DATA_ROOT + 'rvos/'
video_refytvos_image_folder = data_root_refytvos + 'train/JPEGImages/'
video_refytvos_expression_file = data_root_refytvos + 'meta_expressions/train/meta_expressions.json'
video_refytvos_mask_file = data_root_refytvos + 'mask_dict.pkl'

mask2former_processor = dict(
    type=AutoImageProcessor.from_pretrained,
    pretrained_model_name_or_path=mask2former_path,
)


coco_panoseg_dataset = dict(
    type=CoCoPanoSegDataset,
    image_folder=DATA_ROOT+'coco/train2017/',
    pano_gt_folder=DATA_ROOT+'coco/annotations/panoptic_train2017/',
    data_path=DATA_ROOT+'coco/annotations/panoptic_train2017.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    random_ratio=0.6,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)


chn6cug_semseg_dataset = dict(
    type=CHN6CUGSemSegDataset,
    image_folder="./data/koreyoshieo/rs_seg_semantic/chn6-cug/train/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic/chn6-cug/train/mask",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

dgroad_semseg_dataset = dict(
    type=DGROADSemSegDataset,
    image_folder="./data/koreyoshieo/rs_seg_semantic/dg_road/train/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic/dg_road/train/mask",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

evlab_semseg_dataset = dict(
    type=EVLabSemSegDataset,
    image_folder="./data/koreyoshieo/rs_seg_semantic_s2/evlab/train/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic_s2/evlab/train/mask",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

lrsny_semseg_dataset = dict(
    type=LRSNYSemSegDataset,
    image_folder="./data/koreyoshieo/rs_seg_semantic_s2/lrsny/train/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic_s2/lrsny/train/mask",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

oem_semseg_dataset = dict(
    type=OEMSemSegDataset,
    image_folder="./data/koreyoshieo/rs_seg_semantic/oem/train/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic/oem/train/mask",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

ottawa_semseg_dataset = dict(
    type=OTTAWASemSegDataset,
    image_folder="./data/koreyoshieo/rs_seg_semantic_s2/ottawa/train/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic_s2/ottawa/train/mask",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

summer_semseg_dataset = dict(
    type=SUMMERSemSegDataset,
    image_folder="./data/koreyoshieo/rs_seg_semantic_s2/summer/train/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic_s2/summer/train/mask",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

love_da_semseg_dataset = dict(
    type=LoveDASemSegDataset,
    image_folder="./data/koreyoshieo/rs_seg_semantic_s2/lda/train/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic_s2/lda/train/mask",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

bsb_insseg_dataset = dict(
    type=BSBInsSegDataset,
    image_folder="./data/koreyoshieo/rs_seg_panoptic/bsb/image_train_jpg",
    data_path="./data/koreyoshieo/rs_seg_panoptic/bsb/annotations/instance_train.json",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    random_ratio=0.6,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
) # 3,000

nwpu_insseg_dataset = dict(
    type=NWPUInsSegDataset,
    image_folder="./data/koreyoshieo/rs_seg_panoptic/nwpu/train_img",
    data_path="./data/koreyoshieo/rs_seg_panoptic/nwpu/annotations/NWPU_instances_train.json",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    random_ratio=0.6,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
) # 520

# finegrip_panoseg_dataset = dict(
#     type=FineGripPanoSegDataset,
#     image_folder=DATA_ROOT+'koreyoshieo/rs_seg_panoptic/finegrip/Images/Train_panoptic/',
#     pano_gt_folder=DATA_ROOT+'koreyoshieo/rs_seg_panoptic/finegrip/Annotations/Train_panoptic/',
#     data_path=DATA_ROOT+'koreyoshieo/rs_seg_panoptic/finegrip/Annotations/Train_panoptic.json',
#     tokenizer=tokenizer,
#     prompt_template=prompt_template,
#     special_tokens=special_tokens,
#     arch_type='internvl',
#     lazy=True,
#     max_length=max_length,
#     repeats=1,
#     random_ratio=0.6,
#     num_m2f_queries=num_m2f_queries,
#     num_m2f_proposals=num_m2f_proposals,
#     m2f_input_size=1024,
#     m2f_processor=mask2former_processor,
# ) # 901


whumix_insseg_dataset = dict(
    type=WHUMIXInsSegDataset,
    image_folder='./data/whumix_jpg',
    data_path='./data/whumix/train.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    random_ratio=0.6,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

# global_scale_semseg_dataset = dict(
#     type=GlobalScaleDataset,
#     image_folder='/media/home/zhouyikang/datasets/SAMRS/Global-Scale/train',
#     semseg_gt_folder='/media/home/zhouyikang/datasets/SAMRS/Global-Scale/train',
#     tokenizer=tokenizer,
#     prompt_template=prompt_template,
#     special_tokens=special_tokens,
#     arch_type='internvl',
#     lazy=True,
#     max_length=max_length,
#     repeats=1,
#     num_m2f_queries=num_m2f_queries,
#     num_m2f_proposals=num_m2f_proposals,
#     m2f_input_size=1024,
#     m2f_processor=mask2former_processor,
# )

samrs_fast_insseg_dataset = dict(
    type=SAMRSDataset,
    image_folder='./data/samrs/fast/images',
    data_path='./data/samrs/fast/ins',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

samrs_sior_insseg_dataset = dict(
    type=SAMRSDataset,
    image_folder='./data/samrs/sior/JPEGImages-trainval',
    data_path='./data/samrs/sior/ins',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

samrs_sota_insseg_dataset = dict(
    type=SAMRSDataset,
    image_folder='./data/samrs/sota/images',
    data_path='./data/samrs/sota/ins',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)

samrs_sotarbb_insseg_dataset = dict(
    type=SAMRSDataset,
    image_folder='./data/samrs/sota_rbb/images',
    data_path='./data/samrs/sota_rbb/ins',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
)


indonesia_farmland_dataset = dict(
    type=IndonesiaDataset,
    image_folder="./data/zhouyik/RS-DATASET/farmland/images",
    semseg_gt_folder="./data/zhouyik/RS-DATASET/farmland/json",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=10,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
    class_name="farmland",
)

indonesia_road_dataset = dict(
    type=IndonesiaDataset,
    image_folder="./data/zhouyik/RS-DATASET/road/images",
    semseg_gt_folder="./data/zhouyik/RS-DATASET/road/json",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=10,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
    class_name="road",
)

indonesia_water_dataset = dict(
    type=IndonesiaDataset,
    image_folder="./data/zhouyik/RS-DATASET/water/images",
    semseg_gt_folder="./data/zhouyik/RS-DATASET/water/json",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=10,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
    class_name="water",
)

indonesia_vegetation_dataset = dict(
    type=IndonesiaDataset,
    image_folder="./data/zhouyik/RS-DATASET/vegetation/images",
    semseg_gt_folder="./data/zhouyik/RS-DATASET/vegetation/json",
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=10,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
    class_name="vegetation",
)

zhejiang_farmland_dataset1 = dict(
    type=WHUMIXInsSegDataset,
    image_folder='./data/zhouyik/RS-DATASET/cut_png/p1',
    data_path='/mnt/bn/xiangtai-training-data-hl/zhouyikang/pano-vlm/data/zhouyik/RS-DATASET/cut_png/p1_annotations.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    random_ratio=0.6,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
    zhejiang=True,
)

zhejiang_farmland_dataset2 = dict(
    type=WHUMIXInsSegDataset,
    image_folder='/mnt/bn/xiangtai-training-data-hl/zhouyikang/pano-vlm/data/zhouyik/RS-DATASET/cut_png/p2',
    data_path='/mnt/bn/xiangtai-training-data-hl/zhouyikang/pano-vlm/data/zhouyik/RS-DATASET/cut_png/p2_annotations.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    random_ratio=0.6,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
    zhejiang=True,
)

zhejiang_farmland_dataset3 = dict(
    type=WHUMIXInsSegDataset,
    image_folder='/mnt/bn/xiangtai-training-data-hl/zhouyikang/pano-vlm/data/zhouyik/RS-DATASET/cut_png/p3',
    data_path='/mnt/bn/xiangtai-training-data-hl/zhouyikang/pano-vlm/data/zhouyik/RS-DATASET/cut_png/p3_annotations.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    random_ratio=0.6,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
    zhejiang=True,
)

zhejiang_farmland_dataset4 = dict(
    type=WHUMIXInsSegDataset,
    image_folder='/mnt/bn/xiangtai-training-data-hl/zhouyikang/pano-vlm/data/zhouyik/RS-DATASET/cut_png/p4',
    data_path='/mnt/bn/xiangtai-training-data-hl/zhouyikang/pano-vlm/data/zhouyik/RS-DATASET/cut_png/p4_annotations.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    arch_type='internvl',
    lazy=True,
    max_length=max_length,
    repeats=1,
    random_ratio=0.6,
    num_m2f_queries=num_m2f_queries,
    num_m2f_proposals=num_m2f_proposals,
    m2f_input_size=1024,
    m2f_processor=mask2former_processor,
    zhejiang=True,
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        # # coco_panoseg_dataset,
        # # semantic segmentation datasets
        chn6cug_semseg_dataset,
        dgroad_semseg_dataset,
        evlab_semseg_dataset,
        lrsny_semseg_dataset,
        oem_semseg_dataset,
        ottawa_semseg_dataset,
        summer_semseg_dataset,
        love_da_semseg_dataset,
        # # global_scale_semseg_dataset,
        # # instance segmentation dataset
        # bsb_insseg_dataset,
        # nwpu_insseg_dataset,
        whumix_insseg_dataset,
        # # samrs_fast_insseg_dataset,
        # # samrs_sior_insseg_dataset,
        # # samrs_sota_insseg_dataset,
        # # samrs_sotarbb_insseg_dataset,
        indonesia_farmland_dataset,
        indonesia_road_dataset,
        indonesia_water_dataset,
        indonesia_vegetation_dataset,
        # zhejiang_farmland_dataset1,
        # zhejiang_farmland_dataset2,
        # zhejiang_farmland_dataset3,
        # zhejiang_farmland_dataset4,
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
    collate_fn=dict(type=video_lisa_collate_fn)
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
