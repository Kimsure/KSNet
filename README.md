# KSNet-for-VSR


The repository contains the official implementation of "Kernel Dimension Matters: to Activate Avaliable Kernels for Real-time Video Super-resolution"

# Introduction

We have updated the core implementation of KSNet. To start up, you can follow the instructions from [BasicSR](https://github.com/XPixelGroup/BasicSR). More detailed introduction will be updated in a few weeks.

[paper](https://dl.acm.org/doi/abs/10.1145/3581783.3611908)

## Installation

Please follow [installation guide.](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md)

## Data Preparation

Please follow [Dataset preparation.](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)

## Usage

Training Commands

Single GPU Training

```
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
python basicsr/train.py -opt options/train/ECBVSR/train_ECBVSR_REDS.yml
```

Distributed Training
```
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES={GPU device id} \
python -m torch.distributed.launch --nproc_per_node={GPU device number} --master_port=4321 basicsr/train.py options/train/ECBVSR/train_ECBVSR_REDS.yml --launcher pytorch
```
or
```
CUDA_VISIBLE_DEVICES={GPU device id} \
./scripts/dist_train.sh {GPU device number} options/train/ECBVSR/train_ECBVSR_REDS.yml
```


## TODOs

- [x] Build baseline based on [BasicSR](https://github.com/XPixelGroup/BasicSR)   
- [x] Implement kernel split module  
- [x] Implement flow-guided DCN module
- [ ] Clean unrelated code
- [ ] Write clear instructions

## Citation

If you find this work helpful, please consider citing:
```
@inproceedings{jin2023kernel,
  title={Kernel dimension matters: To activate available kernels for real-time video super-resolution},
  author={Jin, Shuo and Liu, Meiqin and Yao, Chao and Lin, Chunyu and Zhao, Yao},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={8617--8625},
  year={2023}
}
```
