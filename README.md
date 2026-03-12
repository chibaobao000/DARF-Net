# DARF-Net
DAFR-Net : Lightweight Deformable Adaptive Feature Reconstruction Network for Tiny in Remote Sensing Images Object Detection

1. Required environments:
Linux
Python 3.6+
PyTorch 1.3+
CUDA 9.2+
GCC 5+
MMCV-full
cocoapi-aitod


2.Prepare Dataset:
Download AI-TODv2 dataset; Download LEVIR-Ship dataset.

3.Train and test：
Train aitodv2 dataset:
python tools/train.py configs_ltdnet/ltdnet/LTDNet_AI_TODv2.py
Train LEVIR-Ship dataset:
python tools/train.py configs_ltdnet/ltdnet/LTDNet_LEVIR_Ship.py
The Pretrained models is on the folder work_dirs.

4. Checkpoint Download（checkpoint-299.pth）：
Baidu Pan：https://pan.baidu.com/s/1eyJiSV12hX6gggiuq8-DFA?pwd=uon2 code：t843
