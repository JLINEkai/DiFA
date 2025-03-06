## Controllable HE-to-IHC Translation with Diffusion Pretraining and Feature Alignment (DiFA)
DiFA is a novel framework HE-to-IHC translation method, which consists of DDPM-based multi-domain pretraining and DDIM-based inter-domain latent feature alignment, to overcome the limitation of pixel misalignment between HE and IHC. We derive a deterministic inference process instead of stochasticity diffusion to ensure the spatial consistency between HE and generated IHC. DiFA achieves controllable and high-quality one-to-many translation with class guidance.

## Install
<!-- 1. Clone this repository: -->

<!-- ```bash
git clone https://github.com/JLINEkai/DiFA.git
cd DiFA
``` -->

Create the conda environment and install the package

```bash
conda create -n difa python=3.10 
conda activate difa
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install timm diffusers accelerate
```


## Training
### Dataset
MIST dataset can be available from from [Google Drive](https://drive.google.com/drive/folders/146V99Zv1LzoHFYlXvSDhKmflIL-joo6p?usp=sharing) and [Baidu Cloud](https://pan.baidu.com/s/1wWlt6tUv4u8bMWU99dj-5g) (code: 6pme).
### Preparation
Extract MIST features using VAE
```bash
torchrun --nnodes=1 --nproc_per_node=1  extract_features.py  --features-path ./mist
```

### Pretraining from Scratch
To launch DiT-S/2 (256x256) training with with `1` GPUs on one node:
```bash
accelerate launch  --mixed_precision fp16 train.py --model DiT-S/2
```

To launch DiT-S/2 (256x256) training with with `8` GPUs on one node:
```bash
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train.py --model DiT-S/2 
```

### Alignment
```bash
accelerate launch --multi_gpu --num_processes 8 train_ddim_cycle.py --model DiT-S/2 
```

## Sampling
```bash
torchrun --nproc_per_node=1 sample_ddim_ddp.py
```

## Visualiztion
<p align="center">
    <img src="visualization.png" width="90%"> <br>
 
  *Comparing the capabilities of SOTA generation methods with our method on the MIST dataset. The differences are highlighted with red boxes*
</p>