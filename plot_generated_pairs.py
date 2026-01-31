import os
import glob
import argparse
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import math

def plot_generated_pairs(input_folder, output_file='generated_samples_viz.png', max_samples=None):
    """
    Reads *_facies.tif and *_ai.tif pairs from input_folder and plots them 
    in a grid format (Facies side-by-side with AI).
    """
    
    # 1. Find files
    facies_files = sorted(glob.glob(os.path.join(input_folder, "*_facies.tif")))
    
    if len(facies_files) == 0:
        print(f"No *_facies.tif files found in {input_folder}")
        return

    samples = []
    
    print(f"Found {len(facies_files)} facies files.")
    
    for f_path in facies_files:
        ai_path = f_path.replace("_facies.tif", "_ai.tif")
        
        if not os.path.exists(ai_path):
            print(f"Warning: AI file not found for {f_path}, skipping.")
            continue
            
        facie = tif.imread(f_path)
        ai = tif.imread(ai_path)
        
        # Squeeze if needed (H, W, 1) -> (H, W) or (1, H, W)
        if facie.ndim == 3: facie = facie.squeeze()
        if ai.ndim == 3: ai = ai.squeeze()
        
        # Basename for title
        name = os.path.basename(f_path).replace("_facies.tif", "")
        samples.append({'name': name, 'facies': facie, 'ai': ai})

    if max_samples is not None and len(samples) > max_samples:
        print(f"Limiting to first {max_samples} samples.")
        samples = samples[:max_samples]
        
    num_samples = len(samples)
    if num_samples == 0:
        print("No complete pairs found.")
        return
    
    nrows = num_samples
    ncols = 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 3 * nrows))
    
    # Handle single sample case (axes is 1D)
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
        
    print("Plotting...")
    
    for i, sample in enumerate(samples):
        # Facies - Col 0
        ax_f = axes[i, 0]
        ax_f.imshow(sample['facies'], cmap='YlGn')
        ax_f.set_title(f"{sample['name']} (Facies)")
        ax_f.axis('off')
        
        # AI - Col 1
        ax_ai = axes[i, 1]
        im = ax_ai.imshow(sample['ai'], cmap='jet')
        ax_ai.set_title(f"{sample['name']} (AI)")
        ax_ai.axis('off')
        
        plt.colorbar(im, ax=ax_ai, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot generated facies/AI pairs.")
    parser.add_argument("--input_folder", required=True, help="Folder containing *_facies.tif and *_ai.tif files")
    parser.add_argument("--output_file", default="generated_samples_viz.png", help="Output PNG file path")
    parser.add_argument("--max_samples", type=int, default=20, help="Max samples to plot (default 20)")
    
    args = parser.parse_args()
    
    plot_generated_pairs(args.input_folder, args.output_file, args.max_samples)
