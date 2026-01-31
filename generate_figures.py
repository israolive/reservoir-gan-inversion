import os
import random
import torch
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import argparse
import json
import gzip
from types import SimpleNamespace

from facies_dataset import FaciesDataset
from models.facies_gan import FaciesGAN
from config import OPT_FILE
from utils import torch2np

def plot_custom_facies(
        fake_facies: list[torch.Tensor],
        real_facies: torch.Tensor,
        masks: torch.Tensor,
        well_indices: list[int],
        fig_idx: int,
        out_dir: str,
        ip_range: tuple = (0, 1)
    ) -> None:
    """
    Custom plotter for 3 wells + 3 realizations.
    """
    num_real_facies = real_facies.size(0) # Should be 3
    num_generated_per_real = fake_facies[0].shape[0] # Should be 3
    num_channels = real_facies.size(1)
    
    # Grid: Rows = Wells, Cols = (Real, Gen1, Gen2, Gen3) * Channels
    total_cols = (num_generated_per_real + 1) * num_channels
    
    # Increase figure width to accommodate colorbars if needed, or just layout
    fig, axes = plt.subplots(num_real_facies, total_cols, figsize=(3.5 * total_cols, num_real_facies * 3.5))
    
    if num_real_facies == 1:
        axes = np.expand_dims(axes, 0)
        
    fake_facies_np = [torch2np(fake_facie, denormalize=True) for fake_facie in fake_facies]
    real_facies_np = torch2np(real_facies, denormalize=True, ceiling=True)
    masks_np = torch2np(masks)
    
    # Round Facies Channel (Channel 0)
    real_facies_np[..., 0] = np.round(real_facies_np[..., 0])
    for i in range(len(fake_facies_np)):
        fake_facies_np[i][..., 0] = np.round(fake_facies_np[i][..., 0])
        
    # Denormalize IP Channel (Channel 1)
    ip_min, ip_max = ip_range
    real_facies_np[..., 1] = real_facies_np[..., 1] * (ip_max - ip_min) + ip_min
    for i in range(len(fake_facies_np)):
        fake_facies_np[i][..., 1] = fake_facies_np[i][..., 1] * (ip_max - ip_min) + ip_min
    
    # Keep track of mappables for colorbar
    im_facies = None
    im_ip = None
    
    for i in range(num_real_facies):
        well_idx = well_indices[i]
        
        # Plot Real
        for c in range(num_channels):
            ax = axes[i, c]
            img = real_facies_np[i, ..., c]
            
            if c == 0:
                cmap = 'YlGn' # Real Facies uses original palette
                vmin, vmax = 0, 1
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                if im_facies is None: im_facies = im
                channel_name = "Facies"
            else:
                cmap = 'jet'
                vmin, vmax = ip_min, ip_max
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                if im_ip is None: im_ip = im
                channel_name = "IP"

            ax.set_title(f'Well {well_idx} {channel_name}')
            
            if c == 0:
                # Plot mask using utils.plot_mask style
                mask_slice = masks_np[i].squeeze()
                mask_sum = np.sum(mask_slice, axis=0) 
                mask_index = np.where(mask_sum == np.max(mask_sum))[0] 
                
                if len(mask_index) > 0:
                    real_vals = real_facies_np[i, ..., 0]
                    x_vals = np.stack([np.full((mask_slice.shape[0],), idx) for idx in mask_index]).flatten()
                    y_vals = np.stack([np.arange(0, mask_slice.shape[0]) for _ in mask_index]).flatten()
                    c_vals = np.stack([real_vals[:, idx] >= 0.5 for idx in mask_index]).flatten().astype(np.int8)
                    ax.scatter(x_vals, y_vals, c=c_vals, s=1, marker='s', cmap='plasma') # Color doesn't matter much if c is binary? 

            ax.axis('off')
            
        # Plot Generated
        for j in range(num_generated_per_real):
            for c in range(num_channels):
                col_idx = (j + 1) * num_channels + c
                ax = axes[i, col_idx]
                img_fake = fake_facies_np[i][j][..., c]
                
                if c == 0:
                    cmap = 'gray' # Generated Facies uses Black/White
                    vmin, vmax = 0, 1
                    im = ax.imshow(img_fake, cmap=cmap, vmin=vmin, vmax=vmax)
                    channel_name = "Facies"
                else:
                    cmap = 'jet'
                    vmin, vmax = ip_min, ip_max
                    im = ax.imshow(img_fake, cmap=cmap, vmin=vmin, vmax=vmax)
                    channel_name = "IP"
                
                ax.set_title(f'Sample {j+1} {channel_name}')
                
                if c == 0:
                     mask_slice = masks_np[i].squeeze()
                     mask_sum = np.sum(mask_slice, axis=0)
                     mask_index = np.where(mask_sum == np.max(mask_sum))[0]
                     
                     if len(mask_index) > 0:
                        real_vals = real_facies_np[i, ..., 0] 
                        x_vals = np.stack([np.full((mask_slice.shape[0],), idx) for idx in mask_index]).flatten()
                        y_vals = np.stack([np.arange(0, mask_slice.shape[0]) for _ in mask_index]).flatten()
                        c_vals = np.stack([real_vals[:, idx] >= 0.5 for idx in mask_index]).flatten().astype(np.int8)
                        ax.scatter(x_vals, y_vals, c=c_vals, s=1, marker='s', cmap='plasma')

                ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(right=0.9) # Make space for colorbars on right
    
    # Add Colorbars
    # IP Colorbar (Continuous)
    cbar_ax_ip = fig.add_axes([0.92, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
    fig.colorbar(im_ip, cax=cbar_ax_ip, label='Acoustic Impedance')
    
    # Add Colorbars
    
    # Facies Colorbar (Discrete) - positioned to the left of IP
    cbar_ax_facies = fig.add_axes([0.89, 0.15, 0.015, 0.7]) 
    cbar_facies = fig.colorbar(im_facies, cax=cbar_ax_facies, ticks=[0, 1])
    cbar_facies.ax.set_yticklabels(['0', '1'])
    cbar_facies.set_label('Facies')
    
    # IP Colorbar (Continuous) - positioned to the right
    cbar_ax_ip = fig.add_axes([0.93, 0.15, 0.015, 0.7]) 
    fig.colorbar(im_ip, cax=cbar_ax_ip, label='Acoustic Impedance')
    
    save_path = os.path.join(out_dir, f'figure_{fig_idx+1}_wells_{"_".join(map(str, well_indices))}.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")

def generate_figures(model_path, data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load options
    with open(os.path.join(model_path, OPT_FILE), "r") as f:
        args = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

    # Force data dir override if needed
    args.input_path = data_dir 
    # Check if we need to set acoustic_impedance_path if it fails?
    # FaciesDataset logic handles fallback to 'ip.npy.gz' now.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset
    dataset = FaciesDataset(args, shuffle=False, ceiling=False)
    total_wells = len(dataset)
    print(f"Total wells in dataset: {total_wells}")
    
    # Load Model
    # Need to reconstruct masked_facies structure for FaciesGAN init?
    # Actually FaciesGAN init takes masked_facies just to derive shapes?
    # Or to use it? 
    # Let's see generate.py... it builds masked_facies. 
    # masked_facies is list of tensors.
    
    masked_facies = []
    for i in range(len(dataset.facies_pyramid)):
         masked_facies.append(torch.stack([mask * facie for mask, facie in zip(dataset.masks_pyramid[i], dataset.facies_pyramid[i])], dim=0))
    
    model = FaciesGAN(device, options=args, masked_facies=masked_facies)
    model.load(model_path, load_discriminator=False, load_masked_facies=False)
    model.generator.eval()
    
    # Generate 10 Figures
    # Each figure: 3 Wells
    # For each well: 3 Realizations
    
    all_indices = list(range(total_wells))
    
    for fig_idx in range(30):
        # Select 3 random wells
        selected_wells = random.sample(all_indices, 3)
        
        # Prepare input batch for these 3 wells
        # We need their masks to generate noise
        # And we need their Real Facies/Masks to plot
        
        real_facies_batch = []
        real_masks_batch = []
        
        # Get raw data for plotting
        # Note: dataset facies are normalized [-1, 1]
        for w_idx in selected_wells:
            # We want the HIGHEST scale (last element of pyramid?)
            # dataset.facies_pyramid[-1] is the finest scale
            f = dataset.facies_pyramid[-1][w_idx]
            m = dataset.masks_pyramid[-1][w_idx]
            real_facies_batch.append(f)
            real_masks_batch.append(m)
            
        real_facies_batch = torch.stack(real_facies_batch).to(device)
        real_masks_batch = torch.stack(real_masks_batch).to(device)
        
        # FaciesGAN.get_noise(mask_indexes) 
        # But get_noise generates 1 noise per index.
        # We need 3 realizations per index.
        # So we need to call generator 3 times? Or construct a batch of 9?
        # If we batch 9, it's easier.
        # Batch: [W1_R1, W1_R2, W1_R3, W2_R1..., W3_R3]
        
        expanded_wells = []
        for w in selected_wells:
            expanded_wells.extend([w]*3) # 3 realizations per well
            
        noises = model.get_noise(expanded_wells, rec=True) # rec=True for reconstruction noise? Or random? 
        # User said "realizations". Usually implies random variation but conditioned.
        # If rec=True, it uses z_rec (fixed?).
        # To get variations, maybe rec=False (random z)?
        # Let's use rec=False for variety.
        noises = model.get_noise(expanded_wells, rec=False)
        
        with torch.no_grad():
            # generate
            gen_out = model.generator(noises, model.noise_amp, in_facie=None)
            final_gen = gen_out # (Batch, C, H, W)
            
        print(f"DEBUG: expanded_wells len: {len(expanded_wells)}")
        print(f"DEBUG: final_gen shape: {final_gen.shape}")
        
        # Reshape final_gen to [3, 3, C, H, W] -> [NumWells, NumRealizations, ...]
        final_gen_reshaped = final_gen.view(3, 3, *final_gen.shape[1:]) # (3, 3, C, H, W)
        
        # Convert to list of tensors expected by plotter?
        # plot_custom_facies expects fake_facies as list[torch.Tensor] len=NumWells
        # where each item is (NumRealizations, C, H, W)
        
        fake_facies_list = [final_gen_reshaped[i] for i in range(3)]
        
        plot_custom_facies(
            fake_facies_list,
            real_facies_batch, 
            real_masks_batch, 
            selected_wells,
            fig_idx,
            output_dir,
            ip_range=(dataset.ip_min, dataset.ip_max)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="figures_output")
    args = parser.parse_args()
    
    generate_figures(args.model_path, args.data_dir, args.output_dir)
