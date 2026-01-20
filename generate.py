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
from sklearn.metrics import euclidean_distances

from facies_dataset import FaciesDataset
from log import format_time
from models.facies_gan import FaciesGAN
from config import OPT_FILE
from types import SimpleNamespace

from utils import torch2np
from sklearn.manifold import MDS
from stats_utils import compute_stats, compute_variogram
import csv

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


def plot_mds(fake_facies, mask_indexes, options):

    fake_facies = np.stack(fake_facies, 0) # (N, 1, H, W, C)
    if fake_facies.shape[-1] == 1:
        fake_facies = fake_facies.squeeze(-1) # (N, 1, H, W)
    fake_facies = fake_facies.squeeze(1) # (N, H, W, C) or (N, H, W)
    
    real_facies = dataset.facies_pyramid[-1]
    real_facies = np.reshape(torch2np(real_facies, denormalize=True), [dataset.facies_pyramid[-1].shape[0], -1])
    fake_facies = np.reshape(fake_facies, [len(mask_indexes), -1])

    real_facies_similarities = euclidean_distances(real_facies)
    fake_facies_similarities = euclidean_distances(fake_facies)
    mds = MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=np.random.RandomState(seed=3),
        dissimilarity="precomputed",
        n_jobs=1,
        normalized_stress="auto",
    )
    real_facies_reduced = mds.fit((real_facies_similarities + real_facies_similarities.T) / 2).embedding_
    real_facies_reduced = real_facies_reduced[options.wells]
    fake_facies_reduced = mds.fit((fake_facies_similarities + fake_facies_similarities.T) / 2).embedding_
    plt.scatter(real_facies_reduced[:, 0], real_facies_reduced[:, 1])
    plt.scatter(fake_facies_reduced[:, 0], fake_facies_reduced[:, 1])
    plt.title("MDS Visualization of FaciesGAN generated facies")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.legend(('Real Facies', 'Fake Facies'), loc='upper right')
    plt.show()

    # fake_facies = np.stack(fake_facies, 0).squeeze(-1)
    # real_facies = dataset.facies_pyramid[-1]
    # real_facies = np.reshape(torch2np(real_facies, denormalize=True), [200, -1])
    # fake_facies = np.reshape(fake_facies, [len(mask_indexes), -1])
    #
    # real_facies_similarities = euclidean_distances(real_facies)
    # fake_facies_similarities = euclidean_distances(fake_facies)
    # mds = MDS(
    #     n_components=2,
    #     max_iter=3000,
    #     eps=1e-9,
    #     random_state=np.random.RandomState(seed=3),
    #     dissimilarity="precomputed",
    #     n_jobs=1,
    #     normalized_stress="auto",
    # )
    # real_facies_reduced = mds.fit((real_facies_similarities + real_facies_similarities.T) / 2).embedding_
    # fake_facies_reduced = mds.fit((fake_facies_similarities + fake_facies_similarities.T) / 2).embedding_
    # sc = plt.scatter(real_facies_reduced[:, 0], real_facies_reduced[:, 1])
    # # plt.scatter(fake_facies_reduced[:, 0], fake_facies_reduced[:, 1])
    # plt.title("MDS Visualization of FaciesGAN generated facies")
    # plt.xlabel("MDS Dimension 1")
    # # plt.ylabel("MDS Dimension 2")
    # # plt.legend(('Real Facies', 'Fake Facies'), loc='upper right')
    # import mplcursors
    # cursor = mplcursors.cursor([sc], hover=True)
    #
    # def label_func(sel):
    #     print(sel.index)
    #     sel.annotation.set_text(str(sel.index))
    #
    # cursor.connect("add", label_func)
    # plt.show()
    # pass




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--how_many", help="how many facies", type=int, required=True)
    parser.add_argument("--model_path", help="models path", type=str, required=True)
    parser.add_argument("--out_path", help="path to save the generated facie", type=str)
    parser.add_argument("--use_gpu", help="use available GPU", action="store_true")
    parser.add_argument("--plot_mds", help="plot the multi dimensional scaling", action="store_true")
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
    parser.add_argument(
        "--calc_stats",
        help="Calculate and print statistics (Mean, Std, Variogram) for generated samples",
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
    # if arguments.plot_mds:
    #     options.num_train_facies = len(options.wells)

    dataset: FaciesDataset = FaciesDataset(options, ceiling=False)
    masked_facies = []
    for i in range(len(dataset.facies_pyramid)):
        masked_facies.append(torch.stack([mask * facie
              for mask, facie in zip(dataset.masks_pyramid[i], dataset.facies_pyramid[i])], dim=0))
    faciesGAN = FaciesGAN(args.device, options=args, masked_facies=masked_facies)
    facies, mi = generate_facies(faciesGAN, arguments.how_many, arguments.model_path, args)


    if arguments.plot_mds: plot_mds(facies, mi, args)
    if arguments.plot_well_mask:
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
            fig.savefig(os.path.join(arguments.out_path, f"generated_facie_{i}.tif"))
            plt.close(fig)
    else:
        for i, facie in enumerate(facies, 1):
            # facie is (1, H, W, C)
            facie = facie.squeeze(0) # (H, W, C)
            if facie.shape[-1] == 1:
                facie = facie.squeeze(-1) # (H, W) standard single channel behavior
            else:
                facie = np.transpose(facie, (2, 0, 1)) # (C, H, W)
                
            tif.imwrite(os.path.join(arguments.out_path, f"generated_facie_{i}.tif"), facie)

    print(f"Facies generated at '{os.path.join(arguments.out_path, 'generated_facie_[1, 2, ...].tif')}'.")
    print(f"Total time: {format_time(int(time.time() - start_time))}")

    if arguments.calc_stats:
        print("\nComputing Statistics...")
        
        def process_batch_stats(samples_np, label="Generated"):
            """
            Args:
                samples_np: (N, H, W, C) numpy array, normalized [0,1]
            Returns:
                dict_list: List of stats dicts per sample
                avg_variogram: (lags, gammas)
            """
            stat_results = []
            
            # Variogram Accumulators
            avg_gammas_acc = None
            lags_ret = None
            var_count = 0

            for i, facie in enumerate(samples_np):
                # facie: (H, W, C)
                if facie.shape[-1] < 2:
                    continue
                    
                f_channel = facie[..., 0]
                ai_channel = facie[..., 1]
                
                # Discretize Facies
                f_channel_discrete = np.round(f_channel).astype(int)
                
                # 1. Scalar Stats
                stats = compute_stats(f_channel_discrete, ai_channel)
                stats['Type'] = label
                stats['Sample_ID'] = i
                stat_results.append(stats)
                
                # 2. Variogram
                l, g = compute_variogram(ai_channel, max_lag=20, n_bins=20)
                if avg_gammas_acc is None:
                    avg_gammas_acc = np.zeros_like(g)
                    lags_ret = l
                
                valid_mask = ~np.isnan(g)
                avg_gammas_acc[valid_mask] += g[valid_mask]
                var_count += 1
                
            if avg_gammas_acc is not None and var_count > 0:
                avg_gammas_acc /= var_count
                
            return stat_results, (lags_ret, avg_gammas_acc)

        # --- 1. Generated Data Stats ---
        # facies is list of (1, H, W, C) -> stack to (N, H, W, C)
        gen_data_np = []
        for f in facies:
             # f is (1, H, W, C)
             sq = f.squeeze(0)
             # ensure (H, W, C)
             if sq.shape[-1] == 1: sq = sq.squeeze(-1) # Handle 1ch case gracefully? Stats need 2ch though.
             gen_data_np.append(sq)
        gen_data_np = np.stack(gen_data_np, axis=0)
        
        gen_stats_list, (gen_lags, gen_gammas) = process_batch_stats(gen_data_np, label="Generated")
        
        # --- 2. Real Data Stats ---
        # dataset.facies_pyramid[-1] is (N_real, C, H, W) tensor, normalized [-1, 1] usually?
        # torch2np handles denorm to [0, 1].
        real_tensor = dataset.facies_pyramid[-1] # (N, C, H, W)
        # torch2np expects (C, H, W) or (N, C, H, W)? utils.py says:
        # if len(x.size()) == 4: return np.transpose(x.numpy(), (0, 2, 3, 1)) -> (N, H, W, C)
        real_data_np = torch2np(real_tensor, denormalize=True) # (N, H, W, C)
        
        real_stats_list, (real_lags, real_gammas) = process_batch_stats(real_data_np, label="Real")
        
        # --- 3. Save Scalar Stats to CSV ---
        all_stats = gen_stats_list + real_stats_list
        csv_path = os.path.join(arguments.out_path, 'statistics.csv')
        if all_stats:
            keys = list(all_stats[0].keys())
            # Ensure Type/Sample_ID are first
            keys.remove('Type'); keys.insert(0, 'Type')
            keys.remove('Sample_ID'); keys.insert(1, 'Sample_ID')
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_stats)
            print(f"Statistics saved to {csv_path}")

        # --- 4. Save Variogram to CSV ---
        var_csv_path = os.path.join(arguments.out_path, 'variogram.csv')
        with open(var_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Lag', 'Real_Semivariance', 'Generated_Semivariance'])
            if gen_lags is not None and real_lags is not None:
                for l, r_g, g_g in zip(gen_lags, real_gammas, gen_gammas):
                    writer.writerow([l, r_g, g_g])
        print(f"Variogram data saved to {var_csv_path}")

        # --- 5. Plot Variogram Comparison ---
        if gen_lags is not None and real_lags is not None:
            plt.figure(figsize=(6, 4))
            plt.plot(gen_lags, gen_gammas, label='Generated', marker='o')
            plt.plot(real_lags, real_gammas, label='Real', marker='x')
            plt.xlabel('Lag Distance (pixels)')
            plt.ylabel('Semivariance')
            plt.title('Isotropic Semivariogram Comparison (Acoustic Impedance)')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(arguments.out_path, 'variogram_comparison.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Variogram plot saved to {plot_path}")
