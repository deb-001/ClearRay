"""
Comparison Super-Resolution Models
SRCNN, RealESRGAN, SwinIR, SRDenseNet — loaded from provided .pth files.
"""

import os
import torch
import torch.nn as nn
from PIL import Image

# Import the actual architectures downloaded from their source repositories
from rrdbnet_arch import RRDBNet
from realesrgan_arch import RealESRGANModel
from lornatang_swinir import swinir_default_sr_x4
from srdensenet_arch import SRDenseNet

# ═══════════════════════════════════════════════════════════════════════════════
# 1. SRCNN (Dong et al., ECCV 2014)
# ═══════════════════════════════════════════════════════════════════════════════
class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)), nn.ReLU(True))
        self.map = nn.Sequential(nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)), nn.ReLU(True))
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        return self.reconstruction(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Model Registry & Loading
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_INFO = {
    "srcnn": {
        "name": "SRCNN",
        "fullname": "Super-Resolution CNN",
        "paper": "Dong et al. (Pre-trained)",
        "description": "3-layer CNN with bicubic pre-upscaling",
        "class": SRCNN,
        "pre_upscale": True,
        "is_rgb": False,
        "weight_file": os.path.join("model", "srcnn_x4-T91-7c460643.pth")
    },
    "realesrgan": {
        "name": "ESRGAN",
        "fullname": "ESRGAN x4plus",
        "paper": "Wang et al. (Pre-trained)",
        "description": "Custom RealESRGAN with correct DFO2K architecture and native grayscale support",
        "class": lambda: RealESRGANModel(),
        "pre_upscale": False,
        "is_rgb": False, # The new wrapper handles 1-channel inputs correctly natively
        "weight_file": os.path.join("model", "RealESRGAN_x4-DFO2K-678bf481.pth")
    },
    "srdensenet": {
        "name": "SRDenseNet",
        "fullname": "Super-Resolution DenseNet",
        "paper": "Tong et al. (Pre-trained)",
        "description": "Dense skip connections for SR",
        "class": lambda: SRDenseNet(in_channels=1, out_channels=1, upscale_factor=4),
        "pre_upscale": False,
        "is_rgb": False,
        "weight_file": os.path.join("model", "SRDenseNet_x4-ImageNet-bb28c23d.pth")
    },
    "swinir": {
        "name": "SwinIR",
        "fullname": "Swin Transformer Image Restoration",
        "paper": "Liang et al. (Pre-trained)",
        "description": "Swin Transformer blocks for classical SR",
        "class": lambda: swinir_default_sr_x4(),
        "pre_upscale": False,
        "is_rgb": True,
        "weight_file": os.path.join("model", "SwinIRNet_default_sr_x4-DIV2K-45658a55.pth")
    }
}


def load_all_models(device="cpu"):
    """Load all comparison models with their pretrained weights."""
    models = {}
    base_dir = os.path.dirname(__file__)

    for key, info in MODEL_INFO.items():
        print(f"Loading {info['name']}...")
        model = info["class"]()
        model.to(device)

        weight_path = os.path.join(base_dir, info["weight_file"])
        if os.path.exists(weight_path):
            sd = torch.load(weight_path, map_location=device, weights_only=False)
            
            # Extract state dict robustly
            if 'params_ema' in sd: sd = sd['params_ema']
            elif 'ema_state_dict' in sd: sd = sd['ema_state_dict']
            elif 'state_dict' in sd: sd = sd['state_dict']
            elif 'params' in sd: sd = sd['params']

            # Strip prefixes like 'model.' or 'module.'
            sd = {k.replace('module.', '').replace('model.', ''): v for k, v in sd.items()}

            # RealESRGANModel loads its own weights internally to avoid dictionary mess
            if key == 'realesrgan':
                model.load_weights(weight_path)
            else:
                model.load_state_dict(sd, strict=False)
            info['trained'] = True
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  ✅ {info['name']}: {param_count:,} params loaded from {info['weight_file']}")
        else:
            info['trained'] = False
            print(f"  ❌ {info['name']}: Weight file {info['weight_file']} not found!")

        model.eval()
        models[key] = model

    return models
