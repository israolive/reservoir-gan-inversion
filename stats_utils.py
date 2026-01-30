import numpy as np


def compute_variogram(data, mask=None, max_lag=20, n_bins=20):
    """
    Compute isotropic experimental semivariogram for 2D data.
    
    Args:
        data (np.ndarray): 2D array of values.
        mask (np.ndarray, optional): Boolean mask. Only True values are used.
        max_lag (int): Maximum lag distance to consider (in pixels).
        n_bins (int): Number of bins for lag distances.
        
    Returns:
        lags (np.ndarray): Lag centers.
        gammas (np.ndarray): Semivariogram values.
    """
    rows, cols = data.shape
    y, x = np.indices((rows, cols))
    
    # Flatten
    values = data.flatten()
    coords = np.column_stack((y.flatten(), x.flatten()))
    
    if mask is not None:
        flat_mask = mask.flatten()
        values = values[flat_mask]
        coords = coords[flat_mask]
    
    # We can't compute pairwise distances for full image (too big: 65k*65k).
    # Instead, we sample indices if the image is large, 
    # OR we use a fast FFT-based approach or simplified axis-aligned approach.
    # For accuracy and simplicity on potentially large sets, let's use random sampling 
    # if total pixels > 2000 to keep it interactive.
    
    n_pixels = values.size
    
    # If not enough pixels, return NaNs
    if n_pixels < 2:
        return np.linspace(0, max_lag, n_bins + 1)[:-1], np.full(n_bins, np.nan)

    if n_pixels > 2500: # 50x50
        # Stratified sampling or just random sampling
        idx = np.random.choice(n_pixels, 2500, replace=False)
        values = values[idx]
        coords = coords[idx]
    
    # Pairwise distances
    dists = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
    sq_diff = (values[:, None] - values[None, :]) ** 2
    
    # Upper triangle only
    triu_idx = np.triu_indices_from(dists, k=1)
    dists = dists[triu_idx]
    sq_diff = sq_diff[triu_idx]
    
    # Binning
    bins = np.linspace(0, max_lag, n_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    gammas = np.zeros(n_bins)
    
    for i in range(n_bins):
        range_mask = (dists >= bins[i]) & (dists < bins[i+1])
        if np.any(range_mask):
            gammas[i] = 0.5 * np.mean(sq_diff[range_mask])
        else:
            gammas[i] = np.nan
            
    return bin_centers, gammas


def compute_stats(facies, ai, facies_labels=None):
    """
    Compute Mean and Std for AI channel, globally and per facies.
    
    Args:
        facies (np.ndarray): Facies channel (H, W).
        ai (np.ndarray): Acoustic Impedance channel (H, W).
        facies_labels (list): List of unique facies values to check. 
                              If None, inferred from data.
                              
    Returns:
        dict: Dictionary of statistics.
    """
    stats = {}
    
    # Global
    stats['Global Mean'] = np.mean(ai)
    stats['Global Std'] = np.std(ai)
    
    # Per Facies
    if facies_labels is None:
        facies_labels = np.unique(facies)
        
    for f in facies_labels:
        mask = (facies == f)
        if np.any(mask):
            stats[f'Facies {f} Mean'] = np.mean(ai[mask])
            stats[f'Facies {f} Std'] = np.std(ai[mask])
        else:
            stats[f'Facies {f} Mean'] = np.nan
            stats[f'Facies {f} Std'] = np.nan
            
    return stats
