# Multi-scale Diffusion Denoised Smoothing (NeurIPS 2023)

This repository contains code for the paper
**"Multi-scale Diffusion Denoised Smoothing"** 
by [Jongheon Jeong](https://sites.google.com/view/jongheonj) and [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html). 

**TL;DR**: *Overcoming the robustness-accuracy trade-off by combining smoothed classifiers of different scales.* 

**Abstract:**
Along with recent diffusion models, randomized smoothing has become one of a few tangible approaches that offers 
adversarial robustness to models at scale, e.g., those of large pre-trained models. 
Specifically, one can perform randomized smoothing on any classifier via a simple "denoise-and-classify" pipeline, 
so-called denoised smoothing, given that an accurate denoiser is available - such as diffusion model. 
In this paper, we present scalable methods to address the current trade-off between certified robustness and accuracy in denoised smoothing. 
Our key idea is to "selectively" apply smoothing among multiple noise scales, coined multi-scale smoothing, 
which can be efficiently implemented with a single diffusion model. This approach also suggests a new objective to compare 
the collective robustness of multi-scale smoothed classifiers, and questions which representation of diffusion model would maximize the objective. 
To address this, we propose to further fine-tune diffusion model (a) to perform consistent denoising whenever the original image is recoverable, 
but (b) to generate rather diverse outputs otherwise. Our experiments show that the proposed multi-scale smoothing scheme, 
combined with diffusion fine-tuning, not only allows strong certified robustness at high noise scales but also maintains accuracy 
close to non-smoothed classifiers.

## Dependencies

[PyTorch](https://pytorch.org/), [Timm](https://github.com/rwightman/pytorch-image-models) and [DeepSpeed](https://github.com/microsoft/DeepSpeed) is needed. CUDA version or GPU difference may slightly influence the results.
```
conda create --yes -n msrs python=3.8
conda activate msrs
bash environments.sh
```

## Scripts

### 1. Fine-tuning CLIP on downstream tasks

```
# Fine-tuning CLIP-B/16 on CIFAR-10. 2 A100-80GB GPUs would work for a run.
DATA=[PATH_TO_DATA] bash scripts/ft_clip/cifar10.sh

# Fine-tuning CLIP-B/16 on ImageNet-1k. 4 A100-80GB GPUs would work for a run.
DATA=[PATH_TO_DATA] bash scripts/ft_clip/imagenet.sh 
```

### 2. Certifying multi-scale smoothed classifiers (CIFAR-10)

#### 2-1. Standard Diffusion-denoised smoothing (sigma = 0.25, 0.5, 1.0; N=10,000)

```
# sigma = 0.25
python certify.py cifar10 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
OUTPUT/certify/cifar10/ft_clip/dds/sigma_0.25_N10k.tsv \
--sigma25 --N=10000 --skip=1 --batch 5000 --arch CLIP_B16 --data_path datasets

# sigma = 0.5
python certify.py cifar10 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
OUTPUT/certify/cifar10/ft_clip/dds/sigma_0.50_N10k.tsv \
--sigma50 --N=10000 --skip=1 --batch 5000 --arch CLIP_B16 --data_path datasets

# sigma = 1.0
python certify.py cifar10 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
OUTPUT/certify/cifar10/ft_clip/dds/sigma_1.00_N10k.tsv \
--sigma100 --N=10000 --skip=1 --batch 5000 --arch CLIP_B16 --data_path datasets
```

#### 2-2. Multi-scale diffusion-denoised smoothing (Ours; sigma = {0.25, 0.5, 1.0}; N=10,000)

```
python certify.py cifar10 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
OUTPUT/certify/cifar10/ft_clip/mdds/sigma_0.25_0.5_1.0_N10k.tsv \
--sigma25 --sigma50 --sigma100 --N=10000 --skip=1 --batch 5000 --arch CLIP_B16 --data_path datasets
```

### 3. Making predictions from multi-scale smoothed classifiers (CIFAR-10)

#### 3-1. Standard Diffusion-denoised smoothing (sigma = 0.25, 0.5, 1.0)

```
# CIFAR-10 
python predict_ddpm.py cifar10 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
0.25 OUTPUT/predict/denoised/ft_clip/dds/cifar10_0.25.tsv --N=200 --skip=1 --batch 200 --arch CLIP_B16 --data_path datasets
python predict_ddpm.py cifar10 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
0.50 OUTPUT/predict/denoised/ft_clip/dds/cifar10_0.50.tsv --N=200 --skip=1 --batch 200 --arch CLIP_B16 --data_path datasets
python predict_ddpm.py cifar10 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
1.00 OUTPUT/predict/denoised/ft_clip/dds/cifar10_1.00.tsv --N=200 --skip=1 --batch 200 --arch CLIP_B16 --data_path datasets

# CIFAR-10-C
python predict_ddpm.py cifar10c OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
0.25 OUTPUT/predict/denoised/ft_clip/dds/cifar10c_0.25 --N=200 --skip=1 --batch 200 --arch CLIP_B16 --data_path datasets
python predict_ddpm.py cifar10c OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
0.50 OUTPUT/predict/denoised/ft_clip/dds/cifar10c_0.50 --N=200 --skip=1 --batch 200 --arch CLIP_B16 --data_path datasets
python predict_ddpm.py cifar10c OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
1.00 OUTPUT/predict/denoised/ft_clip/dds/cifar10c_1.00 --N=200 --skip=1 --batch 200 --arch CLIP_B16 --data_path datasets

# CIFAR-10.1 
python predict_ddpm.py cifar10.1 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
0.25 OUTPUT/predict/denoised/ft_clip/dds/cifar10.1_0.25.tsv --N=200 --skip=1 --batch 200 --arch CLIP_B16 --data_path datasets
python predict_ddpm.py cifar10.1 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
0.50 OUTPUT/predict/denoised/ft_clip/dds/cifar10.1_0.50.tsv --N=200 --skip=1 --batch 200 --arch CLIP_B16 --data_path datasets
python predict_ddpm.py cifar10.1 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
1.00 OUTPUT/predict/denoised/ft_clip/dds/cifar10.1_1.00.tsv --N=200 --skip=1 --batch 200 --arch CLIP_B16 --data_path datasets
```

#### 2-2. Multi-scale diffusion-denoised smoothing (Ours; sigma = {0.25, 0.5, 1.0})

```
# CIFAR-10
python predict_mdds.py cifar10 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
OUTPUT/predict/denoised/ft_clip/mdds/cifar10_n200_0.25_0.50_1.00.tsv --sigma25 --sigma50 --sigma100 --N=200 --skip=1 --batch 200 --skip_p=0.5 --data_path datasets

# CIFAR-10-C
python predict_mdds.py cifar10c OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
OUTPUT/predict/denoised/ft_clip/mdds/cifar10c_n200_0.25_0.50_1.00.tsv --sigma25 --sigma50 --sigma100 --N=200 --skip=1 --batch 200 --skip_p=0.5 --data_path datasets

# CIFAR-10.1
python predict_mdds.py cifar10.1 OUTPUT/CLIP_ft/CLIP_openai/cifar10/FT100_6E4_D06_NC/base/[DATE]/checkpoint-best/mp_rank_00_model_states.pt \
OUTPUT/predict/denoised/ft_clip/mdds/cifar10.1_n200_0.25_0.50_1.00.tsv --sigma25 --sigma50 --sigma100 --N=200 --skip=1 --batch 200 --skip_p=0.5 --data_path datasets
```


# Acknowledgments

This repository is built upon [FT-CLIP](https://github.com/LightDXY/FT-CLIP), which is based on [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [timm](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit) and [CLIP](https://github.com/openai/CLIP) repositories. The CLIP model file is modified from [DeCLIP](https://github.com/Sense-GVT/DeCLIP).


# Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{jeong2023multiscale,
    title={Multi-scale Diffusion Denoised Smoothing},
    author={Jongheon Jeong and Jinwoo Shin},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=zQ4yraDiRe}
}
```


