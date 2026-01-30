import os
import argparse
import numpy as np
import tifffile as tif
import csv
import glob
import gzip
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances

from stats_utils import compute_stats, compute_variogram

def load_generated_samples(input_folder):
    """
    Load generated samples from separate TIF files.
    Expected naming: sample_{i}_facies.tif and sample_{i}_ai.tif
    Returns a list of (H, W, 2) numpy arrays.
    """
    facies_files = sorted(glob.glob(os.path.join(input_folder, "*_facies.tif")))
    
    samples = []
    for f_path in facies_files:
        # Construct corresponding AI path
        # Assuming sample_{i}_facies.tif -> sample_{i}_ai.tif
        ai_path = f_path.replace("_facies.tif", "_ai.tif")
        
        if not os.path.exists(ai_path):
            print(f"Warning: AI file not found for {f_path}, skipping.")
            continue
            
        facie = tif.imread(f_path)
        ai = tif.imread(ai_path)
        
        # Stack them: (H, W, 2)
        # Ensure they are 2D
        if facie.ndim == 3: facie = facie.squeeze()
        if ai.ndim == 3: ai = ai.squeeze()
        
        sample = np.stack([facie, ai], axis=-1)
        samples.append(sample)
        
    return np.array(samples)

def load_real_data_resized(data_dir, target_shape):
    """
    Load raw real data (facies and AI) and resize to target_shape.
    params:
        data_dir: Directory containing facies.npy.gz and acoustic_impedance.npy.gz
        target_shape: Tuple (H, W) for resizing
    returns:
        np.ndarray: (N, target_H, target_W, 2)
    """
    f_path = os.path.join(data_dir, 'facies.npy.gz')
    ai_path = os.path.join(data_dir, 'acoustic_impedance.npy.gz')
    
    # Check existence
    if not os.path.exists(f_path):
        print(f"Error: {f_path} not found.")
        return None
    if not os.path.exists(ai_path):
        print(f"Error: {ai_path} not found.")
        return None
        
    print(f"Loading raw facies from {f_path}")
    with gzip.open(f_path, 'rb') as f:
        facies_raw = np.load(f).astype(np.float32) # (N, H, W) or (N, 1, H, W)
        
    print(f"Loading raw AI from {ai_path}")
    with gzip.open(ai_path, 'rb') as f:
        ai_raw = np.load(f).astype(np.float32) # (N, H, W) or (N, 1, H, W)
        
    # Ensure (N, 1, H, W)
    if facies_raw.ndim == 3: facies_raw = np.expand_dims(facies_raw, axis=1)
    if ai_raw.ndim == 3: ai_raw = np.expand_dims(ai_raw, axis=1)
    
    # Normalize AI to [0, 1]
    ai_min, ai_max = ai_raw.min(), ai_raw.max()
    print(f"Raw AI Min: {ai_min:.2f}, Max: {ai_max:.2f}")
    if ai_max > ai_min:
        ai_raw = (ai_raw - ai_min) / (ai_max - ai_min)
        
    print(f"Resizing real data to {target_shape}...")
    
    # Convert to torch for interpolation
    t_facies = torch.from_numpy(facies_raw)
    t_ai = torch.from_numpy(ai_raw)
    
    # Resize (Bilinear for smooth AI, Bilinear for Facies to match FaciesDataset logic? 
    # Or Nearest for strict categories? FaciesDataset uses Bilinear so we stick to it.)
    t_facies = F.interpolate(t_facies, size=target_shape, mode='bilinear', align_corners=True)
    t_ai = F.interpolate(t_ai, size=target_shape, mode='bilinear', align_corners=True)
    
    # Convert back to numpy and transpose to (N, H, W, C)
    # t_facies: (N, 1, H, W) -> squeeze -> (N, H, W)
    out_facies = t_facies.squeeze(1).numpy()
    out_facies = np.clip(out_facies, 0, None) # Ensure non-negative from interpolation undershoot
    out_ai = t_ai.squeeze(1).numpy()
    
    # Stack -> (N, H, W, 2)
    return np.stack([out_facies, out_ai], axis=-1)


def process_batch_stats(samples_np, label="Generated"):
    """
    Args:
        samples_np: (N, H, W, C) numpy array, normalized [0,1]
    Returns:
        dict_list: List of stats dicts per sample
        dict_variograms: {'Global': (lags, avg_gamma), 'Facies 0': ..., 'Facies 1': ...}
    """
    stat_results = []
    
    # Variogram Accumulators
    # Structure: {'Global': {'acc': zeros, 'count': 0, 'lags': None}, 'Facies 0': ...}
    var_accs = {}

    for i, facie in enumerate(samples_np):
        # facie: (H, W, C)
        if facie.shape[-1] < 2:
            continue
            
        f_channel = facie[..., 0]
        ai_channel = facie[..., 1]
        
        # Discretize Facies (Round float values from resize/generation)
        f_channel_discrete = np.round(f_channel).astype(int)
        
        # 1. Scalar Stats
        stats = compute_stats(f_channel_discrete, ai_channel)
        stats['Type'] = label
        stats['Sample_ID'] = i
        stat_results.append(stats)
        
        # 2. Variograms
        # Identify unique facies in this sample
        unique_facies = np.unique(f_channel_discrete)
        
        # Define tasks: Global + each facies
        tasks = [('Global', None)] + [(f'Facies {f}', (f_channel_discrete == f)) for f in unique_facies]
        
        for name, mask in tasks:
            l, g = compute_variogram(ai_channel, mask=mask, max_lag=20, n_bins=20)
            
            # Init accumulator if new
            if name not in var_accs:
                var_accs[name] = {'acc': np.zeros_like(g), 'count': 0, 'lags': l}
            
            valid_mask = ~np.isnan(g)
            if np.any(valid_mask):
                var_accs[name]['acc'][valid_mask] += g[valid_mask]
                var_accs[name]['count'] += 1
                
    # Average Variograms
    final_variograms = {}
    for name, data in var_accs.items():
        if data['count'] > 0:
            avg_g = data['acc'] / data['count']
            final_variograms[name] = (data['lags'], avg_g)
        
    return stat_results, final_variograms

def plot_mds(real_samples, fake_samples, output_path, wells=None):
    """
    Compute and plot MDS.
    real_samples: (N_real, H, W, C)
    fake_samples: (N_fake, H, W, C)
    """
    # Flatten
    real_flat = real_samples.reshape(real_samples.shape[0], -1)
    fake_flat = fake_samples.reshape(fake_samples.shape[0], -1)
    
    # Subselect real samples if wells provided
    if wells is not None:
        # Check indices validity
        valid_wells = [w for w in wells if w < len(real_flat)]
        if valid_wells:
             real_flat = real_flat[valid_wells]
    
    real_sim = euclidean_distances(real_flat)
    fake_sim = euclidean_distances(fake_flat)
    
    mds = MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=np.random.RandomState(seed=3),
        dissimilarity="precomputed",
        n_jobs=1,
        normalized_stress="auto",
    )
    
    real_emb = mds.fit((real_sim + real_sim.T) / 2).embedding_
    fake_emb = mds.fit((fake_sim + fake_sim.T) / 2).embedding_
    
    plt.figure(figsize=(8, 6))
    plt.scatter(real_emb[:, 0], real_emb[:, 1], label='Real Facies', alpha=0.7)
    plt.scatter(fake_emb[:, 0], fake_emb[:, 1], label='Fake Facies', alpha=0.7)
    plt.title("MDS Visualization")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.legend(loc='upper right')
    plt.savefig(output_path)
    plt.close()
    print(f"MDS plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze generated facies samples.")
    parser.add_argument("--input_folder", required=True, help="Folder containing generated TIF files.")
    parser.add_argument("--real_data_path", required=False, default='data', help="Path to real data folder (containing .npy.gz files). Default 'data'.")
    parser.add_argument("--output_folder", required=False, help="Where to save analysis results. Defaults to input_folder.")
    parser.add_argument("--model_path", required=False, help="Path to model folder (ignored but kept for compatibility).")
    parser.add_argument("--wells", type=int, nargs='+', help="Specific wells indices to use for real data comparison (MDS).")
    
    args = parser.parse_args()
    
    if args.output_folder is None:
        args.output_folder = args.input_folder
        
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 1. Load Generated Data
    print(f"Loading generated samples from {args.input_folder}...")
    gen_data = load_generated_samples(args.input_folder)
    if len(gen_data) == 0:
        print("No generated samples found (looking for *_facies.tif and *_ai.tif). Exiting.")
        return
    print(f"Loaded {len(gen_data)} generated samples. Shape: {gen_data.shape}")
    
    # 2. Load Real Data RESIZED
    target_shape = gen_data.shape[1:3] # (H, W)
    print(f"Target resizing shape: {target_shape}")
    
    print(f"Loading real data from {args.real_data_path}...")
    real_data = load_real_data_resized(args.real_data_path, target_shape)
    
    if real_data is None:
        print("Failed to load real data. Exiting.")
        return
        
    print(f"Loaded real data. Shape: {real_data.shape}")
    
    # 3. Compute Stats
    print(f"DEBUG: Generated Data Shape: {gen_data.shape}")
    print(f"DEBUG: Real Data Shape: {real_data.shape}")
    print("Computing statistics...")
    gen_stats, gen_vars = process_batch_stats(gen_data, label="Generated")
    real_stats, real_vars = process_batch_stats(real_data, label="Real")
    
    # 4. Save CSV
    all_stats = gen_stats + real_stats
    csv_path = os.path.join(args.output_folder, 'statistics.csv')
    if all_stats:
        keys = list(all_stats[0].keys())
        if 'Type' in keys: keys.remove('Type'); keys.insert(0, 'Type')
        if 'Sample_ID' in keys: keys.remove('Sample_ID'); keys.insert(1, 'Sample_ID')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"Statistics saved to {csv_path}")

    # 5. Save Variogram CSV & Plot
    var_csv_path = os.path.join(args.output_folder, 'variogram.csv')
    
    # Collect all variogram keys present in both
    all_var_keys = sorted(list(set(gen_vars.keys()) | set(real_vars.keys())))
    
    with open(var_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['Lag']
        for k in all_var_keys:
            header.append(f'Real_{k}')
            header.append(f'Generated_{k}')
        writer.writerow(header)
        
        # Assuming all lags are same (defined by max_lag/n_bins)
        # We take lags from first available
        if all_var_keys:
             first_key = all_var_keys[0]
             lags = gen_vars.get(first_key, (None, None))[0]
             if lags is None: lags = real_vars.get(first_key, (None, None))[0]
             
             if lags is not None:
                 for i in range(len(lags)):
                     row = [lags[i]]
                     for k in all_var_keys:
                         # Real
                         r_g = real_vars.get(k, (None, None))[1]
                         row.append(r_g[i] if r_g is not None else '')
                         # Gen
                         g_g = gen_vars.get(k, (None, None))[1]
                         row.append(g_g[i] if g_g is not None else '')
                     writer.writerow(row)
                     
    print(f"Variogram data saved to {var_csv_path}")
    
    # Plotting
    # Create a plot for Global and for each Facies
    # Determine unique keys (Global, Facies 0, Facies 1...)
    for k in all_var_keys:
        plt.figure(figsize=(6, 4))
        
        # Generated
        if k in gen_vars:
            l, g = gen_vars[k]
            plt.plot(l, g, label=f'Generated {k}', marker='o')
            
        # Real
        if k in real_vars:
            l, g = real_vars[k]
            plt.plot(l, g, label=f'Real {k}', marker='x')
            
        plt.xlabel('Lag Distance (pixels)')
        plt.ylabel('Semivariance')
        plt.title(f'Isotropic Semivariogram Comparison ({k})')
        plt.legend()
        plt.grid(True)
        filename = f'variogram_comparison_{k.replace(" ", "_").lower()}.png'
        plot_path = os.path.join(args.output_folder, filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Variogram plot saved to {plot_path}")
        
    # 6. MDS Plot
    print("Computing MDS...")
    wells_indices = args.wells
    plot_mds(real_data, gen_data, os.path.join(args.output_folder, 'mds_plot.png'), wells=wells_indices)

if __name__ == "__main__":
    main()
