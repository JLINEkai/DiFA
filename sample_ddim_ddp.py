
    # model = torch.parallel.DistributedDataParallel(model, device_ids=[local_rank])
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
import argparse
import numpy as np
import os
import pickle
# 创建自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, latent_vectors_path, gt_path, path):
        self.latent_vectors = torch.from_numpy(np.load(latent_vectors_path))
        self.gt = torch.from_numpy(np.load(gt_path))
        with open(path, 'rb') as f:
            self.Path = pickle.load(f)
        

    def __len__(self):
        return len(self.latent_vectors)

    def __getitem__(self, idx):
        return self.latent_vectors[idx], self.gt[idx], self.Path[idx]

def main(args):
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    device = f"cuda:{local_rank}"

    # Load model configuration
    latent_size = args.image_size // 8
    
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    # Load checkpoint
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    model_a_params = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(model_a_params['model'], strict=False)

    model.eval()
    # model = torch.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"../sd-vae").to(device)
    ihc_type = args.ihc
    dataset = CustomDataset(f"./mist/MISTcrop1_test_{ihc_type}_he_consistency.npy",
                        f"./mist/MISTcrop1_test_{ihc_type}_ihc_consistency.npy",
                        f"./mist/MISTcrop1_test_{ihc_type}_path_consistency.pkl")
    
    
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    os.makedirs(f"./ddim_mist3_{ihc_type}_r/", exist_ok=True)
    os.makedirs(f"./ddim_mist3_{ihc_type}_s_nofinetuning/", exist_ok=True)
    
    # Batch processing
    start = 0
    for batch_vectors, batch_gt, index in dataloader:
        batch_vectors = batch_vectors.to(device)
        batch_gt = batch_gt.to(device)

        with torch.no_grad():
            model_kwargs = dict(y=torch.from_numpy(np.array([0])).to(device))
            z = diffusion.ddim_reverse_sample_loop(
                model, batch_vectors, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            model_kwargs = dict(y=torch.from_numpy(np.array([2])).to(device)) ############################################ change #######################
            z = diffusion.ddim_sample_loop(
                model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )

            samples = vae.decode(z / 0.18215).sample
            batch_gt = vae.decode(batch_gt / 0.18215).sample
            
        
        # # Save all generated images in the current batch
        for i in range(batch_gt.shape[0]):
            save_image(samples[i:i+1], f"./ddim_mist3_{ihc_type}_s_nofinetuning/{index[i]}", normalize=True, value_range=(-1, 1))
            save_image(batch_gt[i:i+1], f"./ddim_mist3_{ihc_type}_r/{index[i]}", normalize=True, value_range=(-1, 1))


    print("All samples generated and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--ihc", type=str, default="ER") #['HER2', 'ER', 'Ki67', 'PR']
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="./results/006-DiT-S-2/checkpoints/1080000.pt")
    args = parser.parse_args()

    main(args)
    

# torchrun --nproc_per_node=8 --master_port=1235 sample_ddim_ddp.py
# python -m pytorch_fid ./ddim_Ki67_s ./ddim_Ki67_r
# fidelity --gpu 0 --fid --kid --input1 ddim_mist3_HER2_r --input2 ddim_mist3_HER2_s