uv venv /media/disk3/zhouyikang/env/sam3 --python 3.12
source /media/disk3/zhouyikang/env/sam3/bin/activate

uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

cd /media/disk3/zhouyikang/zyk_phd_projects/recognize_geospatial_entity/sam3_main
uv pip install -e .
uv pip install -e ".[train,dev]"

uv pip install einops ninja
uv pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/ronghanghu/cc_torch.git
cd cc_torch/
uv pip install . --no-build-isolation

uv pip install webdataset
uv pip install psutil