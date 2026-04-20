uv pip install torch==2.6.0 torchvision==0.21.0

uv pip install deepspeed==0.17.1

uv pip install triton==3.2.0 accelerate==1.7.0 torchcodec==0.2 peft==0.17.1

uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

cd /media/disk3/zhouyikang
uv pip install -e detectron2 --no-build-isolation

uv pip install xtuner

uv pip install transformers==4.57.0

uv pip install timm

cd /media/disk3/zhouyikang/pano-vlm/projects/llava_dino_m2f/models/ms_deform_attn/ops
uv pip install --no-build-isolation . --verbose
