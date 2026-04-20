from swift.llm import MessagesPreprocessor, DatasetMeta, register_dataset, SubsetDataset, load_dataset
from typing import Any, Callable, Dict, List, Optional, Union
import os
import json
    
register_dataset(
    DatasetMeta(
        dataset_name="mask_generation_rs",
        dataset_path="./data/mask_generation_1024x2_v1.json",
        preprocess_func=MessagesPreprocessor(),
    )
)


