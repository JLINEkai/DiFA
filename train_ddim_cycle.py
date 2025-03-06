# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch
import torch.nn as nn
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from torchvision.utils import save_image
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator
from info_nce import InfoNCE, info_nce
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, noise_he_dir, raw_he_dir, raw_ihc_dir):
        self.noise_he_dir = np.load(noise_he_dir)
        self.raw_he_dir = np.load(raw_he_dir)
        self.raw_ihc_dir = np.load(raw_ihc_dir)

    def __len__(self):
        return self.noise_he_dir.shape[0]

    def __getitem__(self, idx):
        noise_he_dir = self.noise_he_dir[idx]
        raw_he_dir = self.raw_he_dir[idx]
        raw_ihc_dir = self.raw_ihc_dir[idx]

        return torch.from_numpy(noise_he_dir), torch.from_numpy(raw_he_dir), torch.from_numpy(raw_ihc_dir)

infoloss = InfoNCE()
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device
    vae = AutoencoderKL.from_pretrained(f"../sd-vae").to(device)
    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    ckpt_path = args.ckpt
    model_a_params = torch.load(ckpt_path)
    print(model.load_state_dict(model_a_params['model'], strict=False))
    model = model.to(device)
    
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(str(args.num_sampling_steps))  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"../sd-vae").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0)

 
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    
    
    noise_he_dir = "./combined_mist_he_train_ER_consistency.npy"
    raw_he_dir = "./combined_mist_he_train_ER_consistency.npy"
    raw_ihc_dir = "./combined_mist_ihc_train_ER_consistency.npy"
    
    
    dataset = CustomDataset(noise_he_dir, raw_he_dir, raw_ihc_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train() # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for noise_he, raw_he, raw_ihc in loader:
            noise_he = noise_he.squeeze(dim=1)
            raw_he = raw_he.squeeze(dim=1)
            raw_ihc = raw_ihc.squeeze(dim=1)
            
            model_kwargs_he = dict(y=torch.from_numpy(np.array([0])).to(device))
            model_kwargs_ihc = dict(y=torch.from_numpy(np.array([2])).to(device))

            noise_he = noise_he
            with torch.no_grad():
            
                for t in range(230):
                
                    t = torch.from_numpy(np.array([t])).to(device).expand((noise_he.shape[0],))   
                    raw_he = diffusion.ddim_reverse_sample(
                        model, raw_he, t, clip_denoised=False, model_kwargs=model_kwargs_he
                    )['sample']
                    
                    raw_ihc = diffusion.ddim_reverse_sample(
                        model, raw_ihc, t, clip_denoised=False, model_kwargs=model_kwargs_ihc
                    )['sample']
            B = raw_he.shape[0]
            raw_he = torch.cat((raw_he, raw_ihc), dim = 0)    
            model_kwargs = dict(y=torch.cat((torch.from_numpy(np.array([0])).expand((B,)).to(device) , torch.from_numpy(np.array([2])).expand((B,)).to(device)), dim = 0))
                
            for t in range(230, 249):
                t = torch.from_numpy(np.array([t])).to(device).expand((raw_he.shape[0],))

                raw_he = diffusion.ddim_reverse_sample(
                    model, raw_he, t, clip_denoised=False, model_kwargs=model_kwargs
                )['sample']
                
                loss = L1(raw_he[:B], raw_he[B:])
                
                opt.zero_grad()
                accelerator.backward(loss)
                opt.step()
                raw_he  = raw_he.detach()           
            update_ema(ema, model)

            # Log loss values:
            # running_loss += loss.item()
            running_loss += 1
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default="./results/006-DiT-S-2/checkpoints/1080000.pt",
                    help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)



# accelerate launch --multi_gpu --num_processes 8 train_ddim_cycle.py --model DiT-S/2 
