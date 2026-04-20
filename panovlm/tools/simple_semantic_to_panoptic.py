import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
import json
from pycocotools import mask as mask_utils


def entityize_semantic_masks(
    semantic_masks: np.ndarray, 
    class_ids: List[int],
    min_area: int = 0,
    connectivity: int = 8
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    将语义分割masks进行entity化，每个连通区域作为一个独立的entity
    
    Args:
        semantic_masks: 形状为(N, H, W)的numpy数组，每个通道代表一个类别的mask
        class_ids: 对应的类别ID列表，长度应该等于N
        min_area: 最小区域面积阈值，小于此面积的连通区域将被过滤
        connectivity: 连通性类型，4或8
        
    Returns:
        panoptic_mask: 形状为(H, W)的全景分割mask，每个像素值为instance_id
        segments_info: 包含每个instance信息的列表，符合COCO全景分割格式
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
            
            # 添加segment信息，符合COCO全景分割格式
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


def create_panoptic_annotation(
    image_id: int,
    file_name: str,
    height: int,
    width: int,
    panoptic_mask: np.ndarray,
    segments_info: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    创建符合COCO全景分割格式的标注
    
    Args:
        image_id: 图像ID
        file_name: 图像文件名
        height: 图像高度
        width: 图像宽度
        panoptic_mask: 全景分割mask
        segments_info: segment信息列表
        
    Returns:
        COCO格式的全景分割标注
    """
    return {
        'image_id': image_id,
        'file_name': file_name,
        'height': height,
        'width': width,
        'segments_info': segments_info
    }


def save_panoptic_mask(
    panoptic_mask: np.ndarray,
    save_path: str
) -> None:
    """
    保存全景分割mask为图像文件
    
    Args:
        panoptic_mask: 全景分割mask
        save_path: 保存路径
    """
    # 将mask转换为可视化格式
    # 使用不同的颜色来区分不同的instances
    unique_ids = np.unique(panoptic_mask)
    num_instances = len(unique_ids) - 1  # 减去背景
    
    # 创建颜色映射
    np.random.seed(42)
    colors = np.random.randint(0, 255, (num_instances + 1, 3))
    colors[0] = [0, 0, 0]  # 背景为黑色
    
    # 创建彩色mask
    colored_mask = np.zeros((panoptic_mask.shape[0], panoptic_mask.shape[1], 3), dtype=np.uint8)
    
    for i, instance_id in enumerate(unique_ids):
        if instance_id == 0:  # 背景
            continue
        mask = (panoptic_mask == instance_id)
        colored_mask[mask] = colors[i]
    
    # 保存图像
    cv2.imwrite(save_path, colored_mask)
    print(f"全景分割mask已保存到: {save_path}")


def convert_single_image(
    semantic_masks: np.ndarray,
    class_ids: List[int],
    image_id: int,
    file_name: str,
    height: int,
    width: int,
    min_area: int = 0,
    connectivity: int = 8,
    save_mask_path: str = None
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    转换单张图像的语义分割为全景分割
    
    Args:
        semantic_masks: 形状为(N, H, W)的语义分割masks
        class_ids: 对应的类别ID列表
        image_id: 图像ID
        file_name: 图像文件名
        height: 图像高度
        width: 图像宽度
        min_area: 最小区域面积阈值
        connectivity: 连通性类型
        save_mask_path: 保存mask的路径（可选）
        
    Returns:
        panoptic_mask: 全景分割mask
        segments_info: segment信息列表
        annotation: COCO格式的标注
    """
    # 进行entity化
    panoptic_mask, segments_info = entityize_semantic_masks(
        semantic_masks, class_ids, min_area, connectivity
    )
    
    # 创建COCO格式标注
    annotation = create_panoptic_annotation(
        image_id, file_name, height, width, panoptic_mask, segments_info
    )
    
    # 可选：保存mask
    if save_mask_path:
        save_panoptic_mask(panoptic_mask, save_mask_path)
    
    return panoptic_mask, segments_info, annotation


# 示例使用
def example_usage():
    """
    示例：如何使用entityize_semantic_masks函数
    """
    # 创建示例数据
    H, W = 256, 256
    N = 3  # 3个类别
    
    # 创建示例masks
    semantic_masks = np.zeros((N, H, W), dtype=np.uint8)
    
    # 类别1：两个分离的区域（应该变成2个entities）
    semantic_masks[0, 50:100, 50:100] = 1
    semantic_masks[0, 150:200, 150:200] = 1
    
    # 类别2：一个区域（应该变成1个entity）
    semantic_masks[1, 100:150, 100:150] = 1
    
    # 类别3：三个分离的区域（应该变成3个entities）
    semantic_masks[2, 30:60, 30:60] = 1
    semantic_masks[2, 180:210, 180:210] = 1
    semantic_masks[2, 100:130, 200:230] = 1
    
    class_ids = [1, 2, 3]  # 对应的类别ID
    
    print("原始语义分割masks形状:", semantic_masks.shape)
    print("类别ID:", class_ids)
    
    # 转换为全景分割格式
    panoptic_mask, segments_info = entityize_semantic_masks(
        semantic_masks, class_ids, min_area=100
    )
    
    print(f"\n转换结果:")
    print(f"全景分割mask形状: {panoptic_mask.shape}")
    print(f"总共生成 {len(segments_info)} 个entities")
    
    # 打印每个entity的信息
    for i, info in enumerate(segments_info):
        print(f"Entity {i+1}: 类别{info['category_id']}, 面积{info['area']}, ID{info['id']}")
    
    # 统计每个类别的entity数量
    class_counts = {}
    for info in segments_info:
        class_id = info['category_id']
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    print(f"\n每个类别的entity数量:")
    for class_id, count in class_counts.items():
        print(f"类别 {class_id}: {count} 个entities")
    
    return panoptic_mask, segments_info


if __name__ == "__main__":
    # 运行示例
    panoptic_mask, segments_info = example_usage() 