#Models.py
#Human Scar Detection (Industry Application Project)
#COSC 5437 Neural Networking
#Fangze Zhou & Serban Voinea Gabreanu
#This script contains the model definition for the ModernConvNeXtV2 architecture.

import torch
import torch.nn as nn
import math
from typing import Tuple

#Adds a drop path regularization technique, which makes it so it randomly sets the feature maps to zero, 
#which forces the model to learn more robust and redundant pathways.
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

#Applies layer normalization.
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x):
        return self.ln(x.permute(0,2,3,1)).permute(0,3,1,2)

#Global Response Normalization is usde to normalize features for ConvNeXtV2.
#Helps to increase the feature competition and diveristy by normalizing the feauter maps based on their global statistics.
class GRN(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps
    def forward(self, x):
        gx = torch.sqrt(torch.mean(x ** 2, dim=(2,3), keepdim=True) + self.eps)
        return x * (self.gamma / gx) + self.beta

#Core architecture of teh ConvNeXtV2 model. Has depthwise convolution, layer normalization, and an inverted bottlneck MLP with a GRN layer.
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path_rate=0., mlp_ratio=4.0, use_grn=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm   = LayerNorm2d(dim)
        hidden_dim  = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act      = nn.GELU()
        self.grn      = GRN(hidden_dim) if use_grn else nn.Identity()
        self.pwconv2  = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.drop_path(x) + shortcut
        return x

#Dictionary for architecture configurations for different variants.
#For the dataset size used in this project, small should work the best.
_VARIANTS = {
    "tiny":  ((3, 3, 9, 3),  (96, 192, 384, 768)),
    "small": ((3, 3, 27, 3), (96, 192, 384, 768)),
    "base":  ((3, 3, 27, 3), (128, 256, 512, 1024)),
    "large": ((3, 3, 27, 3), (192, 384, 768, 1536)),
}

#Assembles the ConvNeXtV2 model, and it consits of a stem, four stages of ConvNeXtV2 blocks, and a final classification head.
class ModernConvNeXtV2(nn.Module):
    def __init__(self, num_classes: int, variant: str = "base", drop_path_rate: float = 0.2, in_chans: int = 3, use_grn: bool = True, mlp_ratio: float = 4.0):
        super().__init__()
        depths, dims = _VARIANTS[variant]
        self.variant = variant

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        for stage_idx in range(4):
            stage_blocks = []
            for _ in range(depths[stage_idx]):
                stage_blocks.append(
                    ConvNeXtV2Block(dims[stage_idx], drop_path_rate=dpr[cur], mlp_ratio=mlp_ratio, use_grn=use_grn)
                )
                cur += 1
            
            #This part downsamples between each of the stages.
            if stage_idx < 3:
                self.stages.append(nn.Sequential(
                    *stage_blocks,
                    LayerNorm2d(dims[stage_idx]),
                    nn.Conv2d(dims[stage_idx], dims[stage_idx+1], kernel_size=2, stride=2)
                ))
            else:
                self.stages.append(nn.Sequential(*stage_blocks))

        self.norm_head = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    #Helper function for initializing the weights of the convolutional and linear layers, this uses truncated normal intilization.
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    #Defines the forward pass for feature extraction.
    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = x.mean([-2, -1])
        return self.norm_head(x)

    #Define the full forward pass.
    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)

#Helps to create and initialize the model, which makes the model creation process a bit easier from the training script.
def create_modern_cnn(num_classes: int, variant: str = "base", drop_path_rate: float = 0.2):
    print(f"Creating ModernConvNeXtV2 model | Variant: {variant} | Classes: {num_classes} | Drop Path: {drop_path_rate}")
    return ModernConvNeXtV2(num_classes=num_classes, variant=variant, drop_path_rate=drop_path_rate)