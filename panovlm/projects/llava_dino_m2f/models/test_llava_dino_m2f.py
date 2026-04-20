import argparse
import copy
import os.path as osp
from pathlib import Path
import sys
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

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


def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def main():
    config_file = REPO_ROOT / "projects/llava_dino_m2f/configs/internvl3_dino_m2f_2b.py"
    pth_model = REPO_ROOT / "work_dirs/internvl3_dino_m2f_2b/iter_17000.pth"
    cfg = Config.fromfile(config_file)
    model = BUILDER.build(cfg.model)
    backend = get_file_backend(pth_model)

    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(pth_model)
    else:
        state_dict = guess_load_checkpoint(pth_model)

    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval().to(torch.bfloat16)
    # model.mllm.vision_model.to(torch.bfloat16)

    # candidate_class = ['Passenger-Ship', 'Dry-Cargo-Ship', 'Liquid-Cargo-Ship', 'Engineering-Ship', 'A220', 'other-airplane', 'A321', 'Small-Car', 'Van', 'Dump-Truck', 'Intersection', 'Cargo-Truck', 'Bus', 'Tugboat', 'Motorboat', 'Bridge', 'Tennis-Court', 'Baseball-Field', 'Truck-Tractor', 'Fishing-Boat', 'other-ship', 'ARJ21', 'Boeing737', 'Boeing747', 'other-vehicle', 'Boeing787', 'A330', 'Football-Field', 'Basketball-Court', 'Boeing777', 'Roundabout', 'Warship', 'C919', 'Tractor', 'Excavator', 'A350', 'Trailer']
    candidate_class = ['road']
    candidate_class_names = ""
    for name in candidate_class:
        candidate_class_names += f"<p>{name}</p> [CLS], "
    candidate_class_names = candidate_class_names[:-2]
    question = f"<image>\nSegment from the class prompt: {candidate_class_names}"

    image_path = "/media/disk3/dataset/zipped_formatted_datasets_test_split/LandCoverAI/ai_N-34-140-A-d-4-2_267.png"
    image = Image.open(image_path).convert('RGB')
    chat_output = model.chat(text=question, image=image)
    print("response: ", chat_output['response'])
    print('masks: ', chat_output['masks'].shape)
    print('tags: ', chat_output['tags'])

    output_image = visualize(image, chat_output['masks'], chat_output['tags'])
    output_image.save(REPO_ROOT / 'test_llava_dino_m2f.jpg')


if __name__ == "__main__":
    main()
