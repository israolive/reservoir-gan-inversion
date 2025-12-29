
import numpy as np
import matplotlib.pyplot as plt
import gzip
import os

def plot_samples(num_samples=10, output_file='sample_plot.png'):
    facies_path = 'data/facies.npy.gz'
    ai_path = 'data/acoustic_impedance.npy.gz'
    
    if not os.path.exists(facies_path):
        print(f"Error: {facies_path} not found.")
        return
    if not os.path.exists(ai_path):
        print(f"Error: {ai_path} not found. Please run create_dummy_data.py first.")
        return

    print("Loading data...")
    with gzip.open(facies_path, 'rb') as f:
        facies = np.load(f)
    with gzip.open(ai_path, 'rb') as f:
        ai = np.load(f)
        
    print(f"Facies shape: {facies.shape}")
    print(f"AI shape: {ai.shape}")
    
    # Ensure compatible shapes
    # Facies: (N, 1, H, W)
    # AI: (N, H, W) or (N, 1, H, W)
    if facies.ndim == 4:
        facies = facies.squeeze(1)
    if ai.ndim == 4:
        ai = ai.squeeze(1)
        
    total_samples = facies.shape[0]
    #indices = np.random.choice(total_samples, num_samples, replace=False)
    indices = np.arange(min(num_samples, total_samples))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    
    print(f"Unique Facies Values: {np.unique(facies)}")
    
    print("\n--- Statistics per Facies ---")
    
    vals = np.unique(facies)
    val0 = vals[0]
    val1 = vals[-1]

    mask0 = (facies == val0)
    if np.any(mask0):
        mean0 = np.mean(ai[mask0])
        std0 = np.std(ai[mask0])
        print(f"Facies {val0}: Mean AI = {mean0:.2f}, Std AI = {std0:.2f}")
    else:
        print(f"Facies {val0}: No samples found.")
        
    # Facies 1 (or 1)
    mask1 = (facies == val1)
    if np.any(mask1):
        mean1 = np.mean(ai[mask1])
        std1 = np.std(ai[mask1])
        print(f"Facies {val1}: Mean AI = {mean1:.2f}, Std AI = {std1:.2f}")
    else:
        print(f"Facies {val1}: No samples found.")
        
    print("-----------------------------\n")

    for i, idx in enumerate(indices):
        # Plot Facies
        axes[i, 0].imshow(facies[idx], cmap='YlGn')
        axes[i, 0].set_title(f"Facies Sample {idx}")
        axes[i, 0].axis('off')
        
        # Plot Acoustic Impedance
        im = axes[i, 1].imshow(ai[idx], cmap='jet')
        axes[i, 1].set_title(f"AI Sample {idx}")
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    plot_samples()
