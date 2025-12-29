import os
import gzip

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from ops import generate_scales, mask_resize, facie_resize, np2torch


class FaciesDataset(Dataset):
    """
    A PyTorch Dataset class for loading geological facies and their corresponding masks at multiple scales.

    Attributes:
        data_dir (str): Directory containing the facie files.
        scales_list (list): List of scales for resizing facies and masks.
        facies_pyramid (list): List of tensors containing facies at different scales.
        masks_pyramid (list): List of tensors containing masks at different scales.
        resizers (list): List of torchvision transforms for resizing facies and masks.
        ceiling (bool, optional): Whether to set all positive values to 1. Defaults to False.
    """
    def __init__(self, options, shuffle=False, ceiling=True):
        self.data_dir = options.input_path
        self.scales_list = generate_scales(options)
        self.facies_pyramid = [torch.empty((0, 1, *scale)) for scale in self.scales_list]
        self.masks_pyramid = [torch.empty((0, 1, *scale), dtype=torch.int32) for scale in self.scales_list]
        self.resizers = [transforms.Resize(scale[2:]) for scale in self.scales_list[:-1]]
        self.ceiling = ceiling

        facies = np2torch(np.load(gzip.open(os.path.join(self.data_dir, 'facies.npy.gz'), 'rb')))
        masks = np2torch(np.load(gzip.open(os.path.join(self.data_dir, 'masks.npy.gz'), 'rb')))

        if hasattr(options, 'acoustic_impedance_path') and options.acoustic_impedance_path:
             acoustic_impedance = np.load(gzip.open(options.acoustic_impedance_path, 'rb'))
             
             # Normalize AI to [0, 1] so np2torch can map it to [-1, 1]
             ai_min = acoustic_impedance.min()
             ai_max = acoustic_impedance.max()
             print(f"Loaded Acoustic Impedance: Min={ai_min:.2f}, Max={ai_max:.2f}")
             
             if ai_max > ai_min:
                 acoustic_impedance = (acoustic_impedance - ai_min) / (ai_max - ai_min)
             
             acoustic_impedance = np2torch(acoustic_impedance)

             # Assume acoustic_impedance matches facies dimensions
             # Facies shape: (N, 1, H, W)
             if acoustic_impedance.ndim == 3: # (N, H, W)
                 acoustic_impedance = acoustic_impedance.unsqueeze(1)
             
             facies = torch.cat([facies, acoustic_impedance], dim=1)

        for i, scale in enumerate(self.scales_list):
            self.facies_pyramid[i] = torch.stack([
                facie_resize(facie.unsqueeze(0), scale[2:], self.ceiling) for facie in facies], dim=0).squeeze(1)
            self.masks_pyramid[i] = torch.stack([
                mask_resize(mask.unsqueeze(0), scale[2:]) for mask in masks], dim=0).squeeze(1)

        if shuffle:
            self.shuffle()

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.facies_pyramid[0].shape[0]

    def __getitem__(self, idx):
        """
         Returns the facies and masks at the specified index.

         Args:
             idx (int): Index of the sample to retrieve.

         Returns:
             tuple: Tuple of facies and masks at different scales.
         """
        return tuple(facies[idx] for facies in self.facies_pyramid), tuple(masks[idx] for masks in self.masks_pyramid)

    def shuffle(self):
        """
           Shuffles the dataset.
        """
        idxs = torch.randperm(self.__len__())
        for i in range(len(self.scales_list)):
            self.facies_pyramid[i] = self.facies_pyramid[i][idxs]
            self.masks_pyramid[i] = self.masks_pyramid[i][idxs]