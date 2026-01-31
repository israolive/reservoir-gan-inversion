import os
import glob
import gzip
import argparse
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import scipy.stats as st
import re

def compute_histogram(data, bins, density=True):
    hist, _ = np.histogram(data, bins=bins, density=density)
    return hist

def load_data(data_dir):
    print(f"Loading real data from {data_dir}...")
    f_path = os.path.join(data_dir, 'facies.npy.gz')
    
    # Try distinct IP names
    ai_paths = [
        os.path.join(data_dir, 'ip.npy.gz'), 
        os.path.join(data_dir, 'acoustic_impedance.npy.gz')
    ]
    ai_path = None
    for p in ai_paths:
        if os.path.exists(p):
            ai_path = p
            break
            
    m_path = os.path.join(data_dir, 'masks.npy.gz')
    
    if not os.path.exists(f_path): raise FileNotFoundError(f"{f_path} missing")
    if ai_path is None: raise FileNotFoundError("IP/AI file missing")
    if not os.path.exists(m_path): raise FileNotFoundError(f"{m_path} missing")
    
    with gzip.open(f_path, 'rb') as f: facies = np.load(f)
    with gzip.open(ai_path, 'rb') as f: ip = np.load(f)
    with gzip.open(m_path, 'rb') as f: masks = np.load(f)
    
    # Handle dimensions (N, 1, H, W) -> (N, H, W)
    if facies.ndim == 4: facies = facies.squeeze(1)
    if ip.ndim == 4: ip = ip.squeeze(1)
    if masks.ndim == 4: masks = masks.squeeze(1) 
    # Masks might be (N, H, W) or (N, H, W). 
    # Wait, masks usually might be (N, H, W) or (N, C, H, W). Squeeze 1 if present.
    
    print(f"Real Data Shapes: Facies {facies.shape}, IP {ip.shape}, Masks {masks.shape}")
    return facies, ip, masks

def analyze_metrics(input_folder, data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Load Real Data
    real_facies, real_ip, real_masks = load_data(data_dir)
    
    ip_min, ip_max = real_ip.min(), real_ip.max()
    print(f"Real IP Range: [{ip_min:.2f}, {ip_max:.2f}]")
    
    # 2. Load Generated Data
    gen_files = sorted(glob.glob(os.path.join(input_folder, "*_facies.tif")))
    if not gen_files:
        print("No generated files found.")
        return

    # Determine target shape from first generated file
    first_gen = tif.imread(gen_files[0])
    target_shape = first_gen.shape[:2] 
    
    print(f"Target Shape from Generated: {target_shape}")
    
    # Resize Real Data if mismatch
    if real_ip.shape[1:] != target_shape:
        print(f"Resizing Real Data from {real_ip.shape[1:]} to {target_shape}...")
        
        print(f"DEBUG: Unique Real Facies (Pre-Resize): {np.unique(real_facies)}")
        
        # Use torch for easy interpolation as in analyze.py
        import torch
        import torch.nn.functional as F
        
        # Convert to torch (N, 1, H, W)
        t_ip = torch.from_numpy(real_ip).float().unsqueeze(1)
        t_mask = torch.from_numpy(real_masks).float().unsqueeze(1)
        t_facies = torch.from_numpy(real_facies).float().unsqueeze(1)
        
        # Resize
        # IP: Bilinear
        t_ip = F.interpolate(t_ip, size=target_shape, mode='bilinear', align_corners=True)
        # Masks: Nearest or Bilinear? Masks are discrete 0/1 usually. 
        # Ideally masks should be carefully resized. Let's use bilinear then threshold.
        t_mask = F.interpolate(t_mask, size=target_shape, mode='bilinear', align_corners=True)
        # Facies: Nearest (discrete)
        t_facies = F.interpolate(t_facies, size=target_shape, mode='nearest')
        
        real_ip = t_ip.squeeze(1).numpy()
        real_masks = (t_mask.squeeze(1).numpy() > 0.5).astype(int) 
        real_facies = t_facies.squeeze(1).numpy().astype(int)
        
    # Smart Facies Alignment
    # ----------------------
    # Problem: Real labels might be [-1, 1] while Gen are [0, 1].
    # Also, the physical meaning (Low vs High IP) might be inverted in labels.
    # Solution: Align facies based on Mean IP ranking.
    
    unique_real = np.unique(real_facies)
    unique_gen = np.unique([0, 1]) # Assuming generated is binary 0/1
    
    print(f"Aligning Facies Labels...")
    print(f"DEBUG: real_ip shape: {real_ip.shape}")
    print(f"DEBUG: real_facies shape: {real_facies.shape}")
    
    print(f"Aligning Facies Labels...")
    
    unique_real = np.unique(real_facies)
    
    # Compute Mean IP per Real Label
    real_means = []
    for lbl in unique_real:
        mask = (real_facies == lbl)
        mean_val = np.mean(real_ip[mask])
        real_means.append((lbl, mean_val))
    
    # Compute Mean IP per Gen Label (estimate from first batch or all if loaded)
    gen_means = []
    temp_gen_facies = []
    temp_gen_ai = []
    for i in range(min(5, len(gen_files))):
        gf = tif.imread(gen_files[i])
        gai = tif.imread(gen_files[i].replace("_facies.tif", "_ai.tif"))
        gf = np.round(gf).astype(int)
        gai = gai * (ip_max - ip_min) + ip_min # Denormalize estimate
        temp_gen_facies.append(gf.flatten())
        temp_gen_ai.append(gai.flatten())
        
    all_temp_f = np.concatenate(temp_gen_facies)
    all_temp_ai = np.concatenate(temp_gen_ai)
    
    for lbl in unique_gen:
        if np.any(all_temp_f == lbl):
            mean_val = np.mean(all_temp_ai[all_temp_f == lbl])
            gen_means.append((lbl, mean_val))
        else:
            gen_means.append((lbl, 0)) # Fallback
            
    # Sort both by Mean IP
    real_means.sort(key=lambda x: x[1])
    gen_means.sort(key=lambda x: x[1])
    
    print(f"Real Facies Means: {real_means}")
    print(f"Gen Facies Means: {gen_means}")
    
    # Map Rank-to-Rank
    # real_means[0] (Lowest IP) -> gen_means[0] (Lowest IP)
    mapping = {}
    for (r_lbl, _), (g_lbl, _) in zip(real_means, gen_means):
        mapping[r_lbl] = g_lbl
        
    print(f"Facies Mapping (Real -> Gen): {mapping}")
    
    # Apply Mapping
    real_facies_mapped = np.zeros_like(real_facies)
    for r_lbl, g_lbl in mapping.items():
        real_facies_mapped[real_facies == r_lbl] = g_lbl
    real_facies = real_facies_mapped
    
    # Proceed (gen list loop)
    gen_facies_list = []
    gen_ip_list = []
    rmses = []
    
    print(f"Processing {len(gen_files)} generated samples...")
    
    for f_path in gen_files:
        # Expected name: sample_{i}_well_{idx}_facies.tif
        basename = os.path.basename(f_path)
        
        # Parse well index
        match = re.search(r'well_(\d+)_', basename)
        well_idx = int(match.group(1)) if match else None
        
        ai_path = f_path.replace("_facies.tif", "_ai.tif")
        if not os.path.exists(ai_path): continue
        
        g_facie = tif.imread(f_path)
        g_ai = tif.imread(ai_path) # Normalized [0, 1]? Assuming output of generate.py
        
        # Denormalize AI
        # generate.py outputs normalized 0-1? Wait, check utils.py or FaciesGAN
        # generate.py uses torch2np(..., denormalize=True).
        # torch2np maps [-1, 1] -> [0, 1].
        # So g_ai is in [0, 1].
        # We need to map [0, 1] -> [ip_min, ip_max].
        
        g_ai_phys = g_ai * (ip_max - ip_min) + ip_min
        
        # Facies - ensure discrete 0/1 (already rounded in generate.py)
        g_facie_disc = np.round(g_facie).astype(int)
        
        gen_facies_list.append(g_facie_disc)
        gen_ip_list.append(g_ai_phys)
        
        # 3. RMSE Calculation
        if well_idx is not None and well_idx < len(real_masks):
            # Get specific mask
            mask = real_masks[well_idx]
            # Real IP at well locs from Real Image? 
            # Or is real_ip a set of images corresponding to masks?
            # Usually FaciesDataset loads N samples. We generated with indices random.choice(options.wells). 
            # Assuming real_ip[well_idx] is the ground truth for that well.
            
            ground_truth_ip = real_ip[well_idx]
            
            # Mask Location
            # Masks are 1 at wells.
            mask_bool = (mask > 0)
            
            if np.any(mask_bool):
                diff = g_ai_phys[mask_bool] - ground_truth_ip[mask_bool]
                rmse = np.sqrt(np.mean(diff**2))
                rmses.append(rmse)
            else:
                # print(f"Warning: Empty mask for well {well_idx}")
                pass
        else:
            # If filename format old or index out of bounds
            pass

    gen_facies_all = np.concatenate([g.flatten() for g in gen_facies_list])
    gen_ip_all = np.concatenate([g.flatten() for g in gen_ip_list])
    
    real_facies_all = real_facies.flatten()
    real_ip_all = real_ip.flatten()
    
    print("\n--- Computing Metrics ---")
    
    # B. Wasserstein Distance
    print("Computing Wasserstein Distance...")
    # Global
    wd_global = st.wasserstein_distance(real_ip_all, gen_ip_all)
    
    # Per Facies
    real_ip_f0 = real_ip_all[real_facies_all == 0]
    real_ip_f1 = real_ip_all[real_facies_all == 1]
    
    gen_ip_f0 = gen_ip_all[gen_facies_all == 0]
    gen_ip_f1 = gen_ip_all[gen_facies_all == 1]
    
    print(f"Debug Stats: Real F0: {len(real_ip_f0)}, Real F1: {len(real_ip_f1)}")
    print(f"Debug Stats: Gen F0: {len(gen_ip_f0)}, Gen F1: {len(gen_ip_f1)}")
    
    if len(real_ip_f0) > 0 and len(gen_ip_f0) > 0:
        wd_f0 = st.wasserstein_distance(real_ip_f0, gen_ip_f0)
    else:
        wd_f0 = float('nan')
        
    if len(real_ip_f1) > 0 and len(gen_ip_f1) > 0:
        wd_f1 = st.wasserstein_distance(real_ip_f1, gen_ip_f1)
    else:
         wd_f1 = float('nan')
    
    print(f"Wasserstein Distance - Global: {wd_global:.4f}")
    print(f"Wasserstein Distance - Facies 0: {wd_f0:.4f}")
    print(f"Wasserstein Distance - Facies 1: {wd_f1:.4f}")

    # C. RMSE Stats
    if rmses:
        rmse_mean = np.mean(rmses)
        rmse_std = np.std(rmses)
        print(f"RMSE at Wells: Mean={rmse_mean:.4f}, Std={rmse_std:.4f} (N={len(rmses)})")
    else:
        print("Warning: No RMSE calculated (check filename format or masks).")
        rmse_mean, rmse_std = 0.0, 0.0

    # A. Histograms
    print("Plotting Histograms...")
    bins = np.linspace(ip_min, ip_max, 50)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Global
    axes[0].hist(real_ip_all, bins=bins, density=True, alpha=0.5, label='Real', color='blue')
    axes[0].hist(gen_ip_all, bins=bins, density=True, alpha=0.5, label='Generated', color='orange')
    axes[0].set_title(f'Global IP Distribution\nWD={wd_global:.4f}')
    axes[0].legend()
    
    # Facies 0
    axes[1].hist(real_ip_f0, bins=bins, density=True, alpha=0.5, label='Real', color='blue')
    axes[1].hist(gen_ip_f0, bins=bins, density=True, alpha=0.5, label='Generated', color='orange')
    axes[1].set_title(f'IP | Facies 0\nWD={wd_f0:.4f}')
    axes[1].legend()
    
    # Facies 1
    axes[2].hist(real_ip_f1, bins=bins, density=True, alpha=0.5, label='Real', color='blue')
    axes[2].hist(gen_ip_f1, bins=bins, density=True, alpha=0.5, label='Generated', color='orange')
    axes[2].set_title(f'IP | Facies 1\nWD={wd_f1:.4f}')
    axes[2].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'metrics_histograms.png')
    plt.savefig(plot_path)
    print(f"Histogram plot saved to {plot_path}")
    
    # Save Text Report
    with open(os.path.join(output_dir, 'metrics_report.txt'), 'w') as f:
        f.write("Evaluation Metrics Report\n")
        f.write("=========================\n\n")
        f.write(f"Wasserstein Distance (Global): {wd_global:.6f}\n")
        f.write(f"Wasserstein Distance (Facies 0): {wd_f0:.6f}\n")
        f.write(f"Wasserstein Distance (Facies 1): {wd_f1:.6f}\n\n")
        f.write(f"Conditioning RMSE at Wells:\n")
        f.write(f"  Mean: {rmse_mean:.6f}\n")
        f.write(f"  Std:  {rmse_std:.6f}\n")
        f.write(f"  N Samples: {len(rmses)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="analysis_results")
    args = parser.parse_args()
    
    analyze_metrics(args.input_folder, args.data_dir, args.output_dir)
