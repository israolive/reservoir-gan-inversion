import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.custom_layer import ConvBlock
from ops import facie_resize


class Generator(nn.Module):
    """
    A class representing the Generator models.

    Args:
        num_layer (int): Number of layers in the generator.
        kernel_size (int): Size of the convolutional kernel.
        padding_size (int): Size of the padding.
        in_channel (int): Number of input channels (noise + mask).
        out_channel (int): Number of output channels (generated image).
    """
    def __init__(self,
                 num_layer: int,
                 kernel_size: int,
                 padding_size: int,
                 in_channel: int,
                 out_channel: int):
        super(Generator, self).__init__()

        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.zero_padding = self.num_layer * math.floor(self.kernel_size / 2)
        self.full_zero_padding = 2 * self.zero_padding
        self.gens = nn.ModuleList()

    def forward(
            self,
            z: list[torch.Tensor],
            amp: list[float],
            in_facie: torch.Tensor = None,
            start_scale: int = 0,
            stop_scale: int = None,
    ) -> torch.Tensor:
        """
        Forward pass for the generator.

        Args:
            z (list[torch.Tensor]): List of noise tensors for each scale.
            amp (list[float]): List of amplitude values for each scale.
            in_facie (torch.Tensor, optional): Input facie tensor. Defaults to None.
            start_scale (int, optional): Starting scale index. Defaults to 0.
            stop_scale (int, optional): Stopping scale index. Defaults to None.

        Returns:
            torch.Tensor: Output facie tensor.
        """
        if in_facie is None:
            channels = self.out_channel
            height, width = (dim - self.full_zero_padding for dim in z[start_scale].shape[2:])
            in_facie = torch.zeros((z[start_scale].shape[0], channels, height, width), device=z[start_scale].device)

        stop_scale = stop_scale if stop_scale is not None else len(self.gens) - 1

        for index in range(start_scale, stop_scale + 1):
            in_facie = facie_resize(
                in_facie,
                (
                    z[index].shape[2] - self.full_zero_padding,
                    z[index].shape[3] - self.full_zero_padding,
                ),
            )
            
            # Simple noise addition: z * amp
            z_in = z[index] * amp[index]
            
            # Pad in_facie (spatially) first
            in_facie_padded = F.pad(in_facie, [self.zero_padding] * 4, value=0)
            
            # Expand in_facie channels to match z_in (repeat 2 times if z_in is 2*C)
            # Assuming z_in has in_channel channels, and in_facie has out_channel.
            if self.in_channel > self.out_channel:
                 repeat_factor = self.in_channel // self.out_channel
                 in_facie_expanded = in_facie_padded.repeat(1, repeat_factor, 1, 1)
                 # @todo Handle reminder
                 if self.in_channel % self.out_channel != 0:
                     diff = self.in_channel - in_facie_expanded.shape[1]
            else:
                 in_facie_expanded = in_facie_padded

            z_in = z_in + in_facie_expanded

            in_facie = self.gens[index](z_in) + in_facie
        return in_facie

    def create_scale(self, num_feature: int, min_num_feature: int) -> None:
        """
        Create a new scale for the generator.

        Args:
            num_feature (int): The number of features for the convolutional layers.
            min_num_feature (int): The minimum number of features for the convolutional layers.
        """
        head = ConvBlock(self.in_channel, num_feature, self.kernel_size, self.padding_size, 1)
        body = nn.Sequential()

        channels = min_num_feature
        for i in range(self.num_layer - 2):
            channels = int(num_feature / pow(2, (i + 1)))
            block = ConvBlock(max(2 * channels, min_num_feature),
                max(channels, min_num_feature), self.kernel_size, self.padding_size, 1)
            body.add_module(f"block{i + 1}", block)

        tail = nn.Sequential(
            nn.Conv2d(
                max(channels, min_num_feature),
                self.out_channel,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding_size,
            ),
            nn.Tanh(),
        )

        self.gens.append(nn.Sequential(head, body, tail))
