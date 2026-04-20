import torch
from PIL import Image
import os
from pathlib import Path
import numpy as np
import tifffile as tf

TMP_MPL_DIR = Path("/tmp/matplotlib")
TMP_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_MPL_DIR))

HF_HOME_DIR = Path("/tmp/huggingface")
HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME_DIR / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_HOME_DIR / f"modules_{os.getpid()}"))

from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

from types import MethodType
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

from types import MethodType
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.data.detection_utils import read_image, _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.utils.visualizer import GenericMask
import matplotlib.colors as mplc
def draw_instance_predictions_cache(self, labels, np_masks, jittering: bool = True):
    """
    Draw instance-level prediction results on an image.

    Args:
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        jittering: if True, in color mode SEGMENTATION, randomly jitter the colors per class
            to distinguish instances from the same class

    Returns:
        output (VisImage): image object with visualizations.
    """
    boxes = None
    scores = None
    classes = None
    keypoints = None

    masks = [GenericMask(x, self.output.height, self.output.width) for x in np_masks]


    if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
        colors = (
            [self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes]
            if jittering
            else [
                tuple(mplc.to_rgb([x / 255 for x in self.metadata.thing_colors[c]]))
                for c in classes
            ]
        )

        alpha = 0.8
    else:
        colors = None
        alpha = 0.5

    self.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    return self.output


def visualize(input_image, cat_masks, tags):
    if tags is None:
        left_tags = [f'{i}' for i in range(len(cat_masks))]
    else:
        left_tags = tags

    unique_tags = list(set(left_tags))
    text_prompt = ','.join(unique_tags)
    metadata = MetadataCatalog.get("__unused_ape_" + text_prompt)
    metadata.thing_classes = unique_tags
    metadata.stuff_classes = unique_tags

    result_masks = cat_masks
    input_image = _apply_exif_orientation(input_image)
    input_image = convert_PIL_to_numpy(input_image, "BGR")
    visualizer = Visualizer(input_image[:, :, ::-1], metadata, instance_mode=ColorMode.IMAGE)
    visualizer.draw_instance_predictions = MethodType(draw_instance_predictions_cache, visualizer)
    vis_output = visualizer.draw_instance_predictions(labels=left_tags, np_masks=result_masks)
    output_image = vis_output.get_image()
    output_image = Image.fromarray(output_image)

    return output_image

def safe_read_tiff(image_file):
    """
    安全地读取TIFF文件，处理GDAL_NODATA标签解析错误
    
    Args:
        image_file: TIFF文件路径
        
    Returns:
        numpy.ndarray: 图像数组
    """
    try:
        tiff_array = tf.imread(image_file)
    except ValueError as e:
        # 处理GDAL_NODATA标签解析错误
        if "is not castable to uint8" in str(e):
            # 使用PIL作为备选方案读取TIFF文件
            try:
                image = Image.open(image_file)
                tiff_array = np.array(image)
            except Exception as pil_error:
                # 如果PIL也失败，尝试使用tifffile但忽略错误
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tiff_array = tf.imread(image_file, ignore_tags=True)
        else:
            raise e
    
    # 处理多通道数组
    if len(tiff_array.shape) == 3 and tiff_array.shape[2] > 3:
        # 多通道图像，取前3个通道
        tiff_array = tiff_array[:, :, :3]
    elif len(tiff_array.shape) == 2:
        # 灰度图像，转换为3通道
        tiff_array = np.stack([tiff_array] * 3, axis=-1)
    
    # 确保数据类型为uint8
    if tiff_array.dtype != np.uint8:
        if tiff_array.max() <= 1.0:
            tiff_array = (tiff_array * 255).astype(np.uint8)
        else:
            tiff_array = tiff_array.astype(np.uint8)
    
    return tiff_array

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
MODEL_PATH = REPO_ROOT / "zhouyik" / "rs_ov_seg" / "pano-seg-0912"
M2F_PROCESSOR_PATH = REPO_ROOT / "facebook" / "mask2former-swin-large-coco-panoptic"
OUTPUT_DIR = REPO_ROOT / "data" / "for_demo_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

path = str(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
use_flash_attn = device.type == "cuda"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=model_dtype,
    low_cpu_mem_usage=True,
    use_flash_attn=use_flash_attn,
    trust_remote_code=True,
).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
generation_max_new_tokens = 1024 if device.type == "cuda" else 256
model.preparing_for_generation(
    tokenizer=tokenizer,
    max_new_tokens=generation_max_new_tokens,
    torch_dtype=model_dtype,
)

# image_path = "/media/disk3/dataset/zipped_formatted_datasets_test_split/FLAIR/flair_test_061742.png"
image_path_list = ['/media/disk3/dataset/zipped_formatted_datasets/BHP_Watertanks_Watertank/3_0000_0001.jpg',
"/media/disk3/dataset/zipped_formatted_datasets/BSB/247.jpg",
"/media/disk3/dataset/zipped_formatted_datasets/NWPU/131.jpg"]
for image_path in image_path_list:
    image_id = os.path.basename(image_path).split('.')[0]
    if image_path.endswith('.tiff'):
        tiff_array = safe_read_tiff(image_path)
        image = Image.fromarray(tiff_array)
        width, height = image.size
    else:
        image = Image.open(image_path)
        width, height = image.size

    # candidate_class = ['Passenger-Ship', 'Dry-Cargo-Ship', 'Liquid-Cargo-Ship', 'Engineering-Ship', 'A220', 'other-airplane', 'A321', 'Small-Car', 'Van', 'Dump-Truck', 'Intersection', 'Cargo-Truck', 'Bus', 'Tugboat', 'Motorboat', 'Bridge', 'Tennis-Court', 'Baseball-Field', 'Truck-Tractor', 'Fishing-Boat', 'other-ship', 'ARJ21', 'Boeing737', 'Boeing747', 'other-vehicle', 'Boeing787', 'A330', 'Football-Field', 'Basketball-Court', 'Boeing777', 'Roundabout', 'Warship', 'C919', 'Tractor', 'Excavator', 'A350', 'Trailer']

    candidate_class = ['building', 'boat', 'car', 'swimming pool', 'basketball-court', 'football-field', 'road', 'water', 'bareland', 'woodland', 'farmland', 'grassland']
    candidate_class_names = ""
    for name in candidate_class:
        candidate_class_names += f"<p>{name}</p> [CLS], "
    candidate_class_names = candidate_class_names[:-2]
    question = f"<image>\nSegment from the class prompt: {candidate_class_names}"


    m2f_processor = AutoImageProcessor.from_pretrained(str(M2F_PROCESSOR_PATH), trust_remote_code=True,)

    with torch.no_grad():
        chat_outputs = model.predict_forward(text=question, image=image, tokenizer=tokenizer, m2f_processor=m2f_processor)
        answer = chat_outputs['prediction']
        masks = chat_outputs['prediction_masks']

        m2f_outputs = chat_outputs['m2f_outputs']

        label_id_to_text = m2f_outputs['label_id_to_text']

        post_m2f_outputs = model.post_process_panoptic_segmentation(
            m2f_outputs['class_queries_logits'],
            m2f_outputs['masks_queries_logits'],
            target_sizes=[(height, width)],
        )

    masks = post_m2f_outputs[0]['masks']
    ids = post_m2f_outputs[0]['ids']
    pano_masks, pano_tags = [], []
    for id, mask in zip(ids, masks):
        pano_masks.append(mask.unsqueeze(0).cpu().numpy())
        pano_tags.append(label_id_to_text[id.item()])
    pano_masks = np.concatenate(pano_masks, axis=0)


    print(f"user: {question}")
    print(f"assistant: {answer}")

    output_image = visualize(image, pano_masks, pano_tags)
    output_image.save(OUTPUT_DIR / f"{image_id}.jpg")
