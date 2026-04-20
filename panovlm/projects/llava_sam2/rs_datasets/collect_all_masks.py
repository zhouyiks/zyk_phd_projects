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
    LandCoverAISemSegDataset,
)


import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from scipy import ndimage

def segment_road_instances(binary_mask):
	"""
	将二值路网Mask分割为独立的道路实例Mask
	:param binary_mask: 0-1 或 0-255 的 numpy array
	:return: instance_label_map (每个像素的值为其实例ID)
	"""
	# 1. 预处理：确保是二值图
	binary = (binary_mask > 0).astype(np.uint8)
	
	# 2. 提取单像素骨架
	skeleton = skeletonize(binary).astype(np.uint8)
	
	# 3. 寻找拓扑断点（度不等于2的点）
	# 使用 3x3 卷积核计算每个像素邻域内有多少个骨架像素
	kernel = np.array([[1, 1, 1],
					   [1, 10, 1],
					   [1, 1, 1]], dtype=np.uint8)
	neighbor_count = cv2.filter2D(skeleton, -1, kernel)
	
	# 在骨架像素中，neighbor_count 的值为:
	# 11: 端点 (1个邻居 + 自身10)
	# 12: 普通路段点 (2个邻居 + 自身10)
	# >12: 交叉口 (3个以上邻居 + 自身10)
	junctions = (neighbor_count > 12) & (skeleton > 0)
	
	# 4. 从骨架中移除交叉口，得到一段段独立的骨架线
	skeleton_broken = skeleton.copy()
	skeleton_broken[junctions] = 0
	
	# 5. 为每一段骨架线分配 ID (连通域标记)
	# 即使是弯路，只要它是连通的，就会被标记为同一个 ID
	structure = np.ones((3, 3), dtype=int)
	markers, num_features = ndimage.label(skeleton_broken, structure=structure)
	
	# 6. 核心：使用分水岭算法将 ID 传播到整个路面
	# 我们希望 ID 沿着 Mask 内部扩张
	# 距离变换：计算 Mask 内每个点到背景的距离，作为分水岭的“地形”
	distance = ndimage.distance_transform_edt(binary)
	
	# 调用分水岭：以 markers 为种子，在 binary 的范围内生长
	# 使用负距离变换作为表面，使得生长更趋向于路面中心线
	labels = watershed(-distance, markers, mask=binary)
	
	return num_features, labels


def mask_split_road(m):
	'''
 	该函数用于将道路的语义标签拆分为实例
	'''
	instance_list = []
	assert len(m.shape) == 2
	num_labels, labels = segment_road_instances(m)
	for i in range(1, num_labels):
		instance_binary_mask = (labels == i).astype(np.uint8)
		instance_list.append(instance_binary_mask)
	if not len(instance_list):
		instance_list.append(np.zeros_like(labels).astype(np.uint8))
	# NOTE 需要list返回值就直接返回 instance_list
	# m = np.concatenate([m[..., np.newaxis] for m in instance_list], axis=-1)
	return instance_list


def mask_split(m, min_area=None):
	'''
 	该函数用于将普通类别（非道路）的语义标签拆分为实例，比如building、water
	min_area 是用于过滤掉过小的实例的阈值，避免分散的噪声点被当作实例
	'''
	instance_list = []
	assert len(m.shape) == 2
	assert min_area is not None, 'min_area 不能为空'
	num_labels, labels = cv2.connectedComponents(m)
	for i in range(1, num_labels):
		instance_binary_mask = (labels == i).astype(np.uint8)
		if np.sum(instance_binary_mask) < min_area:
			continue
		instance_list.append(instance_binary_mask)
	if not len(instance_list):
		instance_list.append(np.zeros_like(labels).astype(np.uint8))
	# NOTE 需要list返回值就直接返回 instance_list
	# m = np.concatenate([m[..., np.newaxis] for m in instance_list], axis=-1)
	return instance_list

from pycocotools import mask as mask_utils
def decode_mask(object_masks, ori_height, ori_width):
    binary_masks = []
    for object_mask in object_masks:
        if isinstance(object_mask, dict):
            if isinstance(object_mask["counts"], list):
                # convert to compressed RLE
                object_mask = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            m = mask_utils.decode(object_mask)
            m = m.astype(np.uint8).squeeze()
        elif object_mask:
            rles = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            rle = mask_utils.merge(rles)
            m = mask_utils.decode(rle).astype(np.uint8).squeeze()
        else:
            m = np.zeros((ori_height, ori_width), dtype=np.uint8)
        binary_masks.append(m)
    return binary_masks

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
    image_folder="./data/koreyoshieo/rs_seg_semantic/dg_road/test/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic/dg_road/test/mask",
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

flair_semseg_dataset = dict(
    type=FlairSemSegDataset,
    image_folder="./data/koreyoshieo_part2/rs_seg_flair/datasets_seg/FLAIR/flair/test/image",
    semseg_gt_folder="./data/koreyoshieo_part2/rs_seg_flair/datasets_seg/FLAIR/flair/test/mask",
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

dlrsd_semseg_dataset = dict(
    type=DLRSDSemSegDataset,
    image_folder="./data2/datasets_seg/DLRSD/dlrsd/test/image",
    semseg_gt_folder="./data2/datasets_seg/DLRSD/dlrsd/test/mask",
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

dglc_semseg_dataset = dict(
    type=DGLCSemSegDataset,
    image_folder="./data2/datasets_seg/DeepGlobe_LandCover/dg_lc/test/image",
    semseg_gt_folder="./data2/datasets_seg/DeepGlobe_LandCover/dg_lc/test/mask",
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
    image_folder="./data/koreyoshieo/rs_seg_semantic_s2/evlab/test/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic_s2/evlab/test/mask",
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

postdam_semseg_dataset = dict(
    type=POTSDAMSemSegDataset,
    image_folder="./data2/datasets_seg/ISPRS_postdam/potsdam/test/image",
    semseg_gt_folder="./data2/datasets_seg/ISPRS_postdam/potsdam/test/mask",
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

landcoverai_semseg_dataset = dict(
    type=LandCoverAISemSegDataset,
    image_folder="./data2/datasets_seg/LandCover.ai/ai_4c/test/image",
    semseg_gt_folder="./data2/datasets_seg/LandCover.ai/ai_4c/test/mask",
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
    image_folder="./data/koreyoshieo/rs_seg_semantic/oem/test/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic/oem/test/mask",
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
    image_folder="./data/koreyoshieo/rs_seg_semantic_s2/lda/test/image",
    semseg_gt_folder="./data/koreyoshieo/rs_seg_semantic_s2/lda/test/mask",
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
    image_folder="/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/formatted_datasets/BSB",
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
    image_folder="/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/formatted_datasets_test_split/NWPU",
    data_path="./data/koreyoshieo/rs_seg_panoptic/nwpu/annotations/NWPU_instances_val.json",
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

finegrip_panoseg_dataset = dict(
    type=FineGripPanoSegDataset,
    image_folder="/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/formatted_datasets_finegrip/FineGrip",
    pano_gt_folder=DATA_ROOT+'koreyoshieo/rs_seg_panoptic/finegrip/Annotations/Train_panoptic/',
    data_path=DATA_ROOT+'koreyoshieo/rs_seg_panoptic/finegrip/Annotations/Train_panoptic.json',
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
) # 901


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


isaid_insseg_dataset = dict(
    type=ISAIDInsSegDataset,
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

samrs_fast_insseg_dataset = dict(
    type=SAMRSDataset,
    image_folder='/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/formatted_datasets/FAST',
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
    image_folder='/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/formatted_datasets/SIOR',
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
    image_folder='/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/formatted_datasets/SOTA',
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

# samrs_sotarbb_insseg_dataset = dict(
#     type=SAMRSDataset,
#     image_folder='/mnt/bn/xiangtai-training-data-video/zhouyikang/pano-vlm/formatted_datasets/SOTA_RBB',
#     data_path='./data/samrs/sota_rbb/ins',
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

if __name__ == "__main__":
    from xtuner.registry import BUILDER
    import tqdm
    from pycocotools import mask as mask_utils
    import json
    import numpy as np
    import uuid
    import json
    import os
	
    
			
		
	
    # dataset = BUILDER.build(samrs_sota_insseg_dataset)
    # all_data_dict = []
    # for idx in tqdm.tqdm(list(range(len(dataset)))[2:]):
    #     data_dict = dataset[idx]
    #     cls_name_2_masks = dict()
    #     for mask, cls_name in zip(data_dict['masks'], data_dict['class_names']):
    #         # if cls_name in ['A20', 'A19', 'A4', 'A18', 'A9', 'A7', 'A11', 'A3', 'A14', 'A17', 'A5', 'A1', 'A15', 'A8', 'A2', 'A10', 'A13', 'A12', 'A16', 'A6']:
    #         #       cls_name = "Airplane"
    #         rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    #         rle["counts"] = rle["counts"].decode("utf-8")
    #         if cls_name not in cls_name_2_masks:
    #             cls_name_2_masks[cls_name] = []
    #         cls_name_2_masks[cls_name].append(rle)
    #     annotation = []
    #     for k, v in cls_name_2_masks.items():
    #         # if k == "Airplane":
    #         #     print("=========>>>>", len(v))
    #         annotation.append({'class_name': k, 'segmentation': v})
    #     ret_data_dict = {'image_file': "SOTA/"+os.path.basename(data_dict['image_file']), 'annotation': annotation}
    #     all_data_dict.append(ret_data_dict)

    # with open('formatted_datasets_pano/SOTA.json', 'w') as f:
    #     json.dump(all_data_dict, f, indent=4)

    # for json_file in os.listdir("formatted_datasets_test_split"):
    #     if not json_file.endswith('.json'):
    #         continue
    #     with open(os.path.join("formatted_datasets_test_split", json_file), 'r') as f:
    #          json_data = json.load(f)
        
    #     json_name = json_file.replace('.json', '')
    #     for item in tqdm.tqdm(json_data):
    #         image_file = item['image_file']
    #         image_name = os.path.basename(image_file)
    #         new_image_file = f"{json_name}/{image_name}"
    #         item.update({"image_file": new_image_file})
        
    #     with open(os.path.join("formatted_datasets_test_split", json_file), 'w') as f:
    #         json.dump(json_data, f)
             

    # count = 0
    # class_names = []
    # for dataset_dict in [samrs_sior_insseg_dataset, samrs_sota_insseg_dataset, samrs_sotarbb_insseg_dataset]:
    #     dataset = BUILDER.build(dataset_dict)
    #     for idx in tqdm.tqdm(list(range(len(dataset)))):
    #         data_dict = dataset[idx]
    #         for cls_name in data_dict['class_names']:
    #             if cls_name not in class_names:
    #                 class_names.append(cls_name)
    # class_names_str = ", ".join(class_names)
    # print(class_names_str)
    # exit(0)

    # for dataset_dict in [
    #     # chn6cug_semseg_dataset, dgroad_semseg_dataset, 
    #     evlab_semseg_dataset, lrsny_semseg_dataset, oem_semseg_dataset, ottawa_semseg_dataset, summer_semseg_dataset, love_da_semseg_dataset]:
    #     dataset = BUILDER.build(dataset_dict)
    #     # if count < 10:
    #     #     count += 1
    #     for idx in tqdm.tqdm(list(range(len(dataset)))):
    #         data_dict = dataset[idx]
    #         anno_list = []
    #         for mask, cls_name in zip(data_dict['masks'], data_dict['class_names']):
    #             if cls_name in ['road', 'railway']:
    #                 broken_masks = mask_split_road(mask)
    #             else:
    #                 broken_masks = mask_split(mask, min_area=3)
    #             for broken_mask in broken_masks:
    #                 rle = mask_utils.encode(np.array(broken_mask[:, :, None], order="F", dtype="uint8"))[0]
    #                 rle["counts"] = rle["counts"].decode("utf-8")
    #                 anno_list.append({'class_name': cls_name, 'segmentation': rle})
    #         ret_data_dict = {'image_file': data_dict['image_file'], 'annotation': anno_list}
    #         file_name = f"{uuid.uuid4().hex[:8]}"
    #         with open(f'./rs_ov_seg_data_v2/{file_name}.json', 'w') as f:
    #             json.dump(ret_data_dict, f, indent=4)

    # for dataset_dict in [bsb_insseg_dataset, nwpu_insseg_dataset, finegrip_panoseg_dataset, whumix_insseg_dataset, samrs_fast_insseg_dataset, samrs_sior_insseg_dataset, samrs_sota_insseg_dataset, samrs_sotarbb_insseg_dataset]:
    #     dataset = BUILDER.build(dataset_dict)
    #     # if count < 10:
    #     #     count += 1
    #     for idx in tqdm.tqdm(list(range(len(dataset)))):
    #         data_dict = dataset[idx]
    #         anno_list = []
    #         for mask, cls_name in zip(data_dict['masks'], data_dict['class_names']):
    #             rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    #             rle["counts"] = rle["counts"].decode("utf-8")
    #             anno_list.append({'class_name': cls_name, 'segmentation': rle})
    #         ret_data_dict = {'image_file': data_dict['image_file'], 'annotation': anno_list}
    #         file_name = f"{uuid.uuid4().hex[:8]}"
    #         with open(f'./rs_ov_seg_data_v2/{file_name}.json', 'w') as f:
    #             json.dump(ret_data_dict, f, indent=4)

    json_file_list = [
        "./godx7/vector_plus_samtok/CASID.json",
        "./godx7/vector_plus_samtok/CHN6-CUG.json",
        "./godx7/vector_plus_samtok/CITY-OSM.json",
        "./godx7/vector_plus_samtok/CrowdAI.json",
        "./godx7/vector_plus_samtok/FineGrip.json",
        "./godx7/vector_plus_samtok/LRSNY.json",
        "./godx7/vector_plus_samtok/Ottawa.json",
        "./godx7/vector_plus_samtok/RefSegRS.json",
        "./godx7/vector_plus_samtok/UDD_5.json",
        "./godx7/vector_plus_samtok/UDD_6.json",
        "./godx7/vector_plus_samtok/Zurich_Summer.json",
    ]

    for json_file in json_file_list:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        for item in tqdm.tqdm(json_data):
            image_file = item['image_path']
            if not os.path.exists(os.path.join("./godx7/vector_plus_samtok/", image_file)):
                print(f"skip " + str(os.path.join("./godx7/vector_plus_samtok/", image_file)))
                continue

            segms = item['segms']
            anno_list = []
            for segms_item in segms:
                class_name = segms_item['class']
                segmentation_list = segms_item['segmentation']

                if len(segmentation_list) == 1:
                    # need to broken
                    height, width = segmentation_list[0]['size']
                    mask = decode_mask(segmentation_list, height, width)[0]
                    if class_name in ['road', 'railway', 'Road']:
                        broken_masks = mask_split_road(mask)
                    else:
                        broken_masks = mask_split(mask, min_area=3)
                    for broken_mask in broken_masks:
                        rle = mask_utils.encode(np.array(broken_mask[:, :, None], order="F", dtype="uint8"))[0]
                        rle["counts"] = rle["counts"].decode("utf-8")
                        anno_list.append({'class_name': class_name, 'segmentation': rle})
                else:
                    for rle in segmentation_list:
                        anno_list.append({'class_name': class_name, 'segmentation': rle})
            ret_data_dict = {'image_file': os.path.join("./godx7/vector_plus_samtok/", image_file), 'annotation': anno_list}
            file_name = f"{uuid.uuid4().hex[:8]}"
            with open(f'./rs_ov_seg_data_v2/{file_name}.json', 'w') as f:
                json.dump(ret_data_dict, f, indent=4)




    # def scandir_generator(path):
    #     with os.scandir(path) as entries:
    #         for entry in entries:
    #             yield entry.name  # 逐个返回文件名，不占用大量内存

    # all_data = []
    # for json_file in scandir_generator('./rs_ov_seg_data_v2'):
    #     if not json_file.endswith('.json'):
    #         continue
    #     json_path = os.path.join('./rs_ov_seg_data_v2', json_file)
    #     print("===========>>>", json_path)
    #     with open(json_path, 'r') as f:
    #         json_data = json.load(f)
    #         all_data.append(json_data)
    
    # with open('./data/rs_ov_seg_data_image_info_v2.json', 'w') as f:
    #     json.dump(all_data, f, indent=4) 

    # with open('./data/rs_ov_seg_data_image_info_v2.json', 'r') as f:
    #     json_data = json.load(f)
    
    # all_items = []
    # for data_dict in tqdm.tqdm(json_data):
    #     image_file = data_dict['image_file']
    #     for anno in data_dict['annotation']:
    #         segm = anno['segmentation']
    #         save_item = {'image_file': image_file, 'segmentation': segm}
    #         all_items.append(save_item)
    
    # with open('./data/rs_ov_seg_samtok_train_data_v2.json', 'w') as f:
    #     json.dump(all_items, f, indent=4)

    # print(f"{len(all_items)} items.")


    # json_file_list = [
    #     "./godx7/vector_plus_samtok/CASID_samtok.json",
    #     "./godx7/vector_plus_samtok/CHN6-CUG_samtok.json",
    #     "./godx7/vector_plus_samtok/CITY-OSM_samtok.json",
    #     "./godx7/vector_plus_samtok/CrowdAI_samtok.json",
    #     "./godx7/vector_plus_samtok/FineGrip_samtok.json",
    #     "./godx7/vector_plus_samtok/LRSNY_samtok.json",
    #     "./godx7/vector_plus_samtok/Ottawa_samtok.json",
    #     "./godx7/vector_plus_samtok/RefSegRS_samtok.json",
    #     "./godx7/vector_plus_samtok/UDD_5_samtok.json",
    #     "./godx7/vector_plus_samtok/UDD_6_samtok.json",
    #     "./godx7/vector_plus_samtok/WHU_MIX_512_samtok.json",
    #     "./godx7/vector_plus_samtok/Zurich_Summer_samtok.json",
    #     ]
    
    # all_items = []
    # for json_file in json_file_list:
    #     with open(json_file, 'r') as f:
    #         json_data = json.load(f)
    #     for item in tqdm.tqdm(json_data):
    #         image_file = item['image_file']
    #         segmentation = item['segmentation']

    #         if not os.path.exists(os.path.join("./godx7/vector_plus_samtok/", image_file)):
    #             print(f"skip " + str(os.path.join("./godx7/vector_plus_samtok/", image_file)))
    #             continue

    #         ret_item = {
    #             "image_file": os.path.join("./godx7/vector_plus_samtok/", image_file),
    #             "segmentation": segmentation
    #         }
    #         all_items.append(ret_item)
    
    # with open('./data/rs_ov_seg_samtok_train_data_part2.json', 'w') as f:
    #     json.dump(all_items, f, indent=4)
    