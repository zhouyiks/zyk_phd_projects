#!/usr/bin/env python3
"""
使用示例：将语义分割masks转换为全景分割格式

这个脚本展示了如何使用entityize_semantic_masks函数将语义分割的masks
转换为全景分割格式，其中每个连通区域都被视为一个独立的entity。
"""

import numpy as np
import cv2
from simple_semantic_to_panoptic import entityize_semantic_masks, save_panoptic_mask


def create_sample_semantic_masks():
    """
    创建示例语义分割masks
    """
    H, W = 300, 400
    N = 4  # 4个类别
    
    # 创建空的masks
    semantic_masks = np.zeros((N, H, W), dtype=np.uint8)
    
    # 类别1：人 - 两个分离的人（应该变成2个entities）
    semantic_masks[0, 50:120, 50:100] = 1   # 第一个人
    semantic_masks[0, 50:120, 250:300] = 1  # 第二个人
    
    # 类别2：车 - 一个区域（应该变成1个entity）
    semantic_masks[1, 150:200, 100:200] = 1
    
    # 类别3：树 - 三个分离的树（应该变成3个entities）
    semantic_masks[2, 20:60, 20:60] = 1     # 第一棵树
    semantic_masks[2, 20:60, 340:380] = 1   # 第二棵树
    semantic_masks[2, 220:260, 180:220] = 1 # 第三棵树
    
    # 类别4：建筑 - 一个大的建筑区域（应该变成1个entity）
    semantic_masks[3, 80:140, 150:280] = 1
    
    return semantic_masks


def main():
    """
    主函数：演示语义分割到全景分割的转换
    """
    print("=== 语义分割到全景分割转换示例 ===\n")
    
    # 1. 创建示例数据
    print("1. 创建示例语义分割masks...")
    semantic_masks = create_sample_semantic_masks()
    class_ids = [1, 2, 3, 4]  # 人、车、树、建筑
    
    print(f"   Masks形状: {semantic_masks.shape}")
    print(f"   类别ID: {class_ids}")
    
    # 2. 显示每个类别的区域数量
    print("\n2. 分析原始语义分割:")
    for i, (mask, class_id) in enumerate(zip(semantic_masks, class_ids)):
        num_components = cv2.connectedComponents(mask.astype(np.uint8))[0] - 1
        area = np.sum(mask > 0)
        print(f"   类别{class_id}: {num_components}个连通区域, 总面积{area}像素")
    
    # 3. 转换为全景分割格式
    print("\n3. 转换为全景分割格式...")
    panoptic_mask, segments_info = entityize_semantic_masks(
        semantic_masks, 
        class_ids, 
        min_area=50,  # 过滤小于50像素的区域
        connectivity=8
    )
    
    # 4. 显示转换结果
    print(f"\n4. 转换结果:")
    print(f"   全景分割mask形状: {panoptic_mask.shape}")
    print(f"   总共生成 {len(segments_info)} 个entities")
    
    # 5. 详细显示每个entity的信息
    print(f"\n5. 每个entity的详细信息:")
    for i, info in enumerate(segments_info):
        print(f"   Entity {i+1}:")
        print(f"     - ID: {info['id']}")
        print(f"     - 类别: {info['category_id']}")
        print(f"     - 面积: {info['area']} 像素")
        print(f"     - 边界框: {info['bbox']}")
    
    # 6. 统计每个类别的entity数量
    print(f"\n6. 每个类别的entity统计:")
    class_counts = {}
    for info in segments_info:
        class_id = info['category_id']
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        print(f"   类别 {class_id}: {count} 个entities")
    
    # 7. 保存可视化结果
    print(f"\n7. 保存可视化结果...")
    save_panoptic_mask(panoptic_mask, "panoptic_mask_example.png")
    
    # 8. 验证结果
    print(f"\n8. 验证转换结果:")
    unique_ids = np.unique(panoptic_mask)
    print(f"   全景mask中的唯一ID: {unique_ids}")
    print(f"   背景像素数量: {np.sum(panoptic_mask == 0)}")
    
    # 验证每个entity的像素数量
    total_pixels = 0
    for info in segments_info:
        entity_id = info['id']
        entity_pixels = np.sum(panoptic_mask == entity_id)
        total_pixels += entity_pixels
        print(f"   Entity {entity_id}: {entity_pixels} 像素")
    
    print(f"   所有entity总像素数: {total_pixels}")
    
    print(f"\n=== 转换完成 ===")
    print(f"原始语义分割: {len(class_ids)} 个类别")
    print(f"转换后全景分割: {len(segments_info)} 个entities")
    
    return panoptic_mask, segments_info


if __name__ == "__main__":
    # 运行示例
    panoptic_mask, segments_info = main() 