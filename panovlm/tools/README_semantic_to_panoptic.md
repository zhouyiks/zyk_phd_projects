# 语义分割到全景分割转换工具

这个工具包提供了将COCO格式的语义分割数据集转换为全景分割格式的功能。核心思想是将每个连通区域都视为一个独立的entity（实例）。

## 功能特点

- **Entity化处理**：将语义分割中的每个连通区域转换为独立的entity
- **COCO格式兼容**：输出符合COCO全景分割标准的数据格式
- **灵活配置**：支持最小面积过滤、连通性类型选择等参数
- **可视化支持**：提供mask可视化和保存功能

## 文件说明

- `simple_semantic_to_panoptic.py`：核心转换函数
- `example_usage.py`：使用示例脚本
- `semantic_to_panoptic.py`：完整功能版本（包含数据集批量转换）

## 核心函数

### `entityize_semantic_masks()`

将语义分割masks转换为全景分割格式。

```python
def entityize_semantic_masks(
    semantic_masks: np.ndarray,  # 形状为(N, H, W)的numpy数组
    class_ids: List[int],        # 对应的类别ID列表
    min_area: int = 0,           # 最小区域面积阈值
    connectivity: int = 8        # 连通性类型，4或8
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
```

**参数说明：**
- `semantic_masks`：形状为(N, H, W)的numpy数组，每个通道代表一个类别的mask
- `class_ids`：对应的类别ID列表，长度必须等于N
- `min_area`：最小区域面积阈值，小于此面积的连通区域将被过滤
- `connectivity`：连通性类型，4表示4连通，8表示8连通

**返回值：**
- `panoptic_mask`：形状为(H, W)的全景分割mask，每个像素值为instance_id
- `segments_info`：包含每个instance信息的列表，符合COCO全景分割格式

## 使用示例

### 基本使用

```python
import numpy as np
from simple_semantic_to_panoptic import entityize_semantic_masks

# 假设您有语义分割masks
semantic_masks = np.zeros((3, 256, 256), dtype=np.uint8)  # 3个类别
class_ids = [1, 2, 3]

# 添加一些示例数据
semantic_masks[0, 50:100, 50:100] = 1  # 类别1的第一个区域
semantic_masks[0, 150:200, 150:200] = 1  # 类别1的第二个区域
semantic_masks[1, 100:150, 100:150] = 1  # 类别2的区域
semantic_masks[2, 30:60, 30:60] = 1      # 类别3的区域

# 转换为全景分割格式
panoptic_mask, segments_info = entityize_semantic_masks(
    semantic_masks, class_ids, min_area=100
)

print(f"生成了 {len(segments_info)} 个entities")
```

### 运行示例脚本

```bash
cd tools
python example_usage.py
```

这将运行一个完整的示例，展示转换过程并生成可视化结果。

## 输出格式

### 全景分割Mask

`panoptic_mask`是一个形状为(H, W)的numpy数组：
- 值为0表示背景
- 值为1, 2, 3, ...表示不同的entity实例

### Segments信息

每个segment包含以下信息：
```python
{
    'id': instance_id,           # 实例ID
    'category_id': class_id,     # 类别ID
    'area': area,               # 区域面积
    'bbox': [x, y, w, h],       # 边界框
    'iscrowd': 0,               # 是否为群体标注
    'segmentation': rle          # RLE编码的mask
}
```

## 转换逻辑

1. **输入**：语义分割masks (N, H, W)，每个通道代表一个类别
2. **连通组件分析**：对每个类别的mask进行连通组件分析
3. **Entity化**：每个连通区域被分配一个唯一的instance_id
4. **过滤**：根据面积阈值过滤小区域
5. **输出**：全景分割mask和segments信息

## 示例转换结果

假设输入有3个类别的语义分割：
- 类别1：2个分离的区域 → 2个entities
- 类别2：1个区域 → 1个entity  
- 类别3：3个分离的区域 → 3个entities

转换后总共生成6个entities，每个都有唯一的instance_id。

## 依赖库

```bash
pip install numpy opencv-python pycocotools matplotlib
```

## 注意事项

1. **连通性选择**：8连通通常比4连通更准确，但计算量稍大
2. **面积过滤**：建议设置合适的min_area值来过滤噪声
3. **内存使用**：对于大图像，注意内存使用情况
4. **类别ID**：确保class_ids列表与masks的顺序一致

## 扩展功能

如果需要批量处理整个数据集，可以使用`semantic_to_panoptic.py`中的`convert_semantic_to_panoptic_dataset()`函数。

## 常见问题

**Q: 如何处理重叠的masks？**
A: 当前实现假设masks不重叠。如果有重叠，后处理的mask会覆盖前面的。

**Q: 如何设置合适的最小面积阈值？**
A: 建议根据图像分辨率和目标对象大小来设置，通常为图像面积的0.1%-1%。

**Q: 输出格式是否兼容COCO评估工具？**
A: 是的，输出格式完全兼容COCO全景分割评估工具。 