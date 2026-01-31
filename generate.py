import copy
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import json
import time
import tifffile as tif

from argparse import ArgumentParser
from matplotlib import pyplot as plt

from facies_dataset import FaciesDataset
from log import format_time
from models.facies_gan import FaciesGAN
from config import OPT_FILE
from types import SimpleNamespace

from utils import torch2np

def generate_facies(model: FaciesGAN, how_many: int, model_path: str, options: SimpleNamespace) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate facies using the FaciesGAN model.

        Args:
            model (FaciesGAN): The FaciesGAN model instance.
            how_many (int): Number of facies to generate.
            model_path (str): Path to the model.
            options (SimpleNamespace): Options for the model.

        Returns:
            Tuple[List[np.ndarray], List[int]]: A tuple containing a list of generated facies as numpy arrays and a list of mask indexes.
        """
        model.load(model_path, load_discriminator=False, load_masked_facies=False)
        model.generator.eval()

        mask_indexes = [random.choice(options.wells) for _ in range(how_many)]

        noises = model.get_noise(mask_indexes, rec=options.rec)

        with torch.no_grad():
            generated_facies = [
                torch2np(gen_facie.unsqueeze(0), denormalize=True)
                for gen_facie in model.generator(noises, model.noise_amp, in_facie=None)
            ]
        return generated_facies, mask_indexes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--how_many", help="how many facies", type=int, required=True)
    parser.add_argument("--model_path", help="models path", type=str, required=True)
    parser.add_argument("--out_path", help="path to save the generated facie", type=str)
    parser.add_argument("--use_gpu", help="use available GPU", action="store_true")
    # plot_mds is removed as analysis is separate, but we keep the arg to avoid breaking scripts if they pass it (ignored)
    parser.add_argument("--plot_mds", help="plot the multi dimensional scaling (Ignored: use analyze.py)", action="store_true")
    parser.add_argument(
        "--wells",
        help="list of well indices to generate facies from",
        type=int,
        nargs='+',
        default=tuple(range(200)),
    )
    parser.add_argument(
        "--rec",
        help="generate a sample with the reconstruction noise. The reconstruction sample will have the same size as the TI",
        action="store_true",
    )
    parser.add_argument("--gpu_device", help="GPU device", type=int, default=0)
    parser.add_argument(
        "--plot_well_mask",
        help="Add/plot also the well masks on each generated facies",
        action="store_true"
    )
    # calc_stats removed effectively
    parser.add_argument(
        "--calc_stats",
        help="Calculate and print statistics (Ignored: use analyze.py)",
        action="store_true"
    )

    arguments = parser.parse_args()

    if arguments.out_path is None:
        arguments.out_path = arguments.model_path

    if not os.path.exists(arguments.out_path):
        os.makedirs(arguments.out_path)

    device = torch.device(f"cuda:{arguments.gpu_device}" if torch.cuda.is_available()
                          else f"mps:{arguments.gpu_device}" if torch.backends.mps.is_available()
                          else f"cpu:{arguments.gpu_device}")

    # Load options
    with open(os.path.join(arguments.model_path, OPT_FILE), "r") as f:
        args = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
        args.use_gpu = arguments.use_gpu
        args.rec = arguments.rec
        args.wells = arguments.wells
        args.device = device
        args.data_dir = 'data'

    start_time = time.time()

    print("Generating facies...")

    options = copy.copy(args)
    
    dataset: FaciesDataset = FaciesDataset(options, ceiling=False)
    masked_facies = []
    
    # We only need masked_facies if we are plotting well masks OR if model needs them?
    # generator inputs: noises (which might use masks if needed? FaciesGAN uses masked_facies for noise gen?)
    # FaciesGAN.get_noise logic: if self.masked_facies is loaded.
    # We passed load_masked_facies=False to model.load().
    # But FaciesGAN init takes masked_facies.
    
    for i in range(len(dataset.facies_pyramid)):
        masked_facies.append(torch.stack([mask * facie
              for mask, facie in zip(dataset.masks_pyramid[i], dataset.facies_pyramid[i])], dim=0))
              
    faciesGAN = FaciesGAN(args.device, options=args, masked_facies=masked_facies)
    facies, mi = generate_facies(faciesGAN, arguments.how_many, arguments.model_path, args)


    if arguments.plot_well_mask:
        print("Plotting well masks...")
        for i, (facie, masked_facie) in enumerate(zip(facies, [masked_facies[-1][i] for i in mi]), 1):
            # facie shape: (1, H, W, C)
            facie = facie.squeeze(0) # (H, W, C)
            num_channels = facie.shape[-1]
            
            masked_facie = np.squeeze(masked_facie.numpy()) # (H, W)? Mask is single channel?
            if masked_facie.ndim == 3:
                 masked_facie = masked_facie[0] # Assume mask is 1 channel
            
            mask_index = np.argmax(np.sum(masked_facie != 0, axis=0))
            
            fig, axes = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4))
            if num_channels == 1:
                axes = [axes]
                
            for c in range(num_channels):
                ax = axes[c]
                img = facie[..., c]
                cmap = 'YlGn' if c == 0 else 'jet' # Facies=YlGn, AI=jet
                
                ax.imshow(img, cmap=cmap)
                ax.set_title(f'Ch {c}')
                
                if c == 0: # Only plot mask on Facies
                    ax.scatter(
                        np.full((masked_facie.shape[0],), mask_index),
                        np.arange(0, masked_facie.shape[0]),
                        c=np.astype(masked_facie[:, mask_index] >= 0.5, np.int8),
                        s=1, marker='s', cmap='plasma', label="Facies Mask"
                    )
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')

            fig.tight_layout()
            # Save visualization
            fig.savefig(os.path.join(arguments.out_path, f"generated_facie_{i}_viz.tif"))
            plt.close(fig)
            
            # ALSO save separate channels for analysis if desired?
            # User said "Do not delete the logic to render well data".
            # The prompt implies that if we are generating for analysis, we might use --plot_well_mask?
            # Or usually we don't.
            # I will ensure we ALWAYS save the raw data split by channel, even if we assume visualization is separate.
            # But duplicate saving might be clutter.
            # I'll save raw data regardless.

    # Always save raw separate channels for analyze.py
    for i, (facie, well_idx) in enumerate(zip(facies, mi), 1):
        # facie is (1, H, W, C)
        facie = facie.squeeze(0) # (H, W, C)
        
        # Ensure it has channels, handle 1CH vs 2CH vs etc
        if facie.ndim == 2: # (H, W) -> (H, W, 1) to make logic uniform
             facie = np.expand_dims(facie, axis=-1)
             
        num_channels = facie.shape[-1]
        
        if num_channels >= 1:
            facies_img = facie[..., 0]
            # Apply rounding to discrete facies
            facies_img = np.round(facies_img).astype(np.uint8)
            tif.imwrite(os.path.join(arguments.out_path, f"sample_{i}_well_{well_idx}_facies.tif"), facies_img)
            
        # Channel 1: Acoustic Impedance
        if num_channels >= 2:
            ai_img = facie[..., 1]
            tif.imwrite(os.path.join(arguments.out_path, f"sample_{i}_well_{well_idx}_ai.tif"), ai_img)

    print(f"Facies generated at '{os.path.join(arguments.out_path, 'sample_[i]_[facies|ai].tif')}'.")
    if arguments.plot_well_mask:
        print(f"Visualizations saved as 'generated_facie_[i]_viz.tif'.")
        
    print(f"Total time: {format_time(int(time.time() - start_time))}")
