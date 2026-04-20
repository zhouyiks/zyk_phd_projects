import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
import json
from pycocotools import mask as mask_utils


def semantic_to_panoptic_masks(
    semantic_masks: np.ndarray, 
    class_ids: List[int],
    min_area: int = 0,
    connectivity: int = 8
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    将语义分割masks转换为全景分割格式
    
    Args:
        semantic_masks: 形状为(N, H, W)的numpy数组，每个通道代表一个类别的mask
        class_ids: 对应的类别ID列表，长度应该等于N
        min_area: 最小区域面积阈值，小于此面积的连通区域将被过滤
        connectivity: 连通性类型，4或8
        
    Returns:
        panoptic_mask: 形状为(H, W)的全景分割mask，每个像素值为instance_id
        segments_info: 包含每个instance信息的列表
    """
    assert semantic_masks.shape[0] == len(class_ids), "masks数量必须等于类别ID数量"
    
    H, W = semantic_masks.shape[1], semantic_masks.shape[2]
    panoptic_mask = np.zeros((H, W), dtype=np.int32)
    segments_info = []
    current_instance_id = 1
    
    for mask_idx, (mask, class_id) in enumerate(zip(semantic_masks, class_ids)):
        # 确保mask是二值的
        binary_mask = (mask > 0).astype(np.uint8)
        
        if np.sum(binary_mask) == 0:
            continue
            
        # 使用OpenCV进行连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=connectivity
        )
        
        # 跳过背景标签（标签0）
        for label_id in range(1, num_labels):
            # 获取当前连通区域的mask
            component_mask = (labels == label_id).astype(np.uint8)
            area = np.sum(component_mask)
            
            # 过滤小区域
            if area < min_area:
                continue
                
            # 将当前instance添加到全景mask中
            panoptic_mask[component_mask > 0] = current_instance_id
            
            # 计算边界框
            y_coords, x_coords = np.where(component_mask > 0)
            if len(y_coords) > 0 and len(x_coords) > 0:
                bbox = [
                    int(np.min(x_coords)),  # x_min
                    int(np.min(y_coords)),  # y_min
                    int(np.max(x_coords) - np.min(x_coords) + 1),  # width
                    int(np.max(y_coords) - np.min(y_coords) + 1)   # height
                ]
            else:
                bbox = [0, 0, 0, 0]
            
            # 计算RLE编码
            rle = mask_utils.encode(np.asfortranarray(component_mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            # 添加segment信息
            segment_info = {
                'id': current_instance_id,
                'category_id': class_id,
                'area': int(area),
                'bbox': bbox,
                'iscrowd': 0,
                'segmentation': rle
            }
            segments_info.append(segment_info)
            
            current_instance_id += 1
    
    return panoptic_mask, segments_info


def convert_semantic_to_panoptic_dataset(
    semantic_annotations: List[Dict[str, Any]],
    output_file: str,
    min_area: int = 0,
    connectivity: int = 8
) -> None:
    """
    将整个语义分割数据集转换为全景分割格式
    
    Args:
        semantic_annotations: 语义分割标注列表，每个元素包含图像信息和masks
        output_file: 输出文件路径
        min_area: 最小区域面积阈值
        connectivity: 连通性类型
    """
    panoptic_annotations = []
    
    for ann in semantic_annotations:
        # 假设每个标注包含以下字段
        image_id = ann['image_id']
        file_name = ann['file_name']
        height = ann['height']
        width = ann['width']
        
        # 获取masks和类别ID
        masks = np.array(ann['masks'])  # 形状应该是(N, H, W)
        class_ids = ann['class_ids']    # 对应的类别ID列表
        
        # 转换为全景分割格式
        panoptic_mask, segments_info = semantic_to_panoptic_masks(
            masks, class_ids, min_area, connectivity
        )
        
        # 创建全景分割标注
        panoptic_ann = {
            'image_id': image_id,
            'file_name': file_name,
            'height': height,
            'width': width,
            'segments_info': segments_info
        }
        
        panoptic_annotations.append(panoptic_ann)
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(panoptic_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"已成功转换 {len(panoptic_annotations)} 个标注到 {output_file}")


def visualize_panoptic_mask(
    panoptic_mask: np.ndarray,
    segments_info: List[Dict[str, Any]],
    image: np.ndarray = None,
    save_path: str = None
) -> np.ndarray:
    """
    可视化全景分割mask
    
    Args:
        panoptic_mask: 全景分割mask
        segments_info: segment信息列表
        image: 原始图像（可选）
        save_path: 保存路径（可选）
        
    Returns:
        可视化结果图像
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # 创建随机颜色映射
    np.random.seed(42)
    num_instances = len(segments_info)
    colors = np.random.randint(0, 255, (num_instances, 3))
    
    # 创建可视化图像
    if image is not None:
        vis_image = image.copy()
    else:
        vis_image = np.zeros((panoptic_mask.shape[0], panoptic_mask.shape[1], 3), dtype=np.uint8)
    
    # 为每个instance着色
    for i, segment_info in enumerate(segments_info):
        instance_id = segment_info['id']
        color = colors[i]
        mask = (panoptic_mask == instance_id)
        vis_image[mask] = color
    
    # 添加边界框和标签
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(vis_image)
    
    for segment_info in segments_info:
        bbox = segment_info['bbox']
        category_id = segment_info['category_id']
        
        rect = Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 5, f'Class {category_id}', 
                color='red', fontsize=8, weight='bold')
    
    ax.set_title('Panoptic Segmentation Visualization')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"可视化结果已保存到 {save_path}")
    
    plt.show()
    return vis_image


# 示例使用函数
def example_usage():
    """
    示例：如何使用这些函数
    """
    # 假设我们有以下语义分割数据
    H, W = 256, 256
    N = 3  # 3个类别
    
    # 创建示例masks
    semantic_masks = np.zeros((N, H, W), dtype=np.uint8)
    
    # 类别1：两个分离的区域
    semantic_masks[0, 50:100, 50:100] = 1
    semantic_masks[0, 150:200, 150:200] = 1
    
    # 类别2：一个区域
    semantic_masks[1, 100:150, 100:150] = 1
    
    # 类别3：三个分离的区域
    semantic_masks[2, 30:60, 30:60] = 1
    semantic_masks[2, 180:210, 180:210] = 1
    semantic_masks[2, 100:130, 200:230] = 1
    
    class_ids = [1, 2, 3]  # 对应的类别ID
    
    # 转换为全景分割格式
    panoptic_mask, segments_info = semantic_to_panoptic_masks(
        semantic_masks, class_ids, min_area=100
    )
    
    print(f"原始语义分割：{N}个类别")
    print(f"转换后全景分割：{len(segments_info)}个instances")
    
    for i, info in enumerate(segments_info):
        print(f"Instance {i+1}: 类别{info['category_id']}, 面积{info['area']}, ID{info['id']}")
    
    return panoptic_mask, segments_info


if __name__ == "__main__":
    # 运行示例
    panoptic_mask, segments_info = example_usage() 