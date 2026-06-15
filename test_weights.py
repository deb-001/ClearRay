import torch
import torch.nn as nn
from rrdbnet_arch import RRDBNet
from network_swinir import SwinIR
from srdensenet_arch import SRDenseNet

device = 'cpu'

# 1. SRCNN
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)), nn.ReLU(True))
        self.map = nn.Sequential(nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)), nn.ReLU(True))
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        return self.reconstruction(out)

srcnn = SRCNN().to(device)
sd = torch.load('model/srcnn_x4-T91-7c460643.pth', map_location=device, weights_only=False)
if 'state_dict' in sd: sd = sd['state_dict']
elif 'params' in sd: sd = sd['params']
srcnn.load_state_dict(sd)
print("SRCNN loaded successfully")

# 2. RealESRGAN
realesrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32).to(device)
sd = torch.load('model/RealESRGAN_x4-DFO2K-678bf481.pth', map_location=device, weights_only=False)
sd = sd.get('params_ema', sd.get('params', sd))
realesrgan.load_state_dict(sd)
print("RealESRGAN loaded successfully")

# 3. SwinIR
swinir = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)
sd = torch.load('model/SwinIRNet_default_sr_x4-DIV2K-45658a55.pth', map_location=device, weights_only=False)
if 'params_ema' in sd: sd = sd['params_ema']
elif 'state_dict' in sd: sd = sd['state_dict']
swinir.load_state_dict(sd, strict=True)
print("SwinIR loaded successfully")

# 4. SRDenseNet
srdensenet = SRDenseNet(in_channels=1, out_channels=1, upscale_factor=4).to(device)
sd = torch.load('model/SRDenseNet_x4-ImageNet-bb28c23d.pth', map_location=device, weights_only=False)
if 'state_dict' in sd: sd = sd['state_dict']
srdensenet.load_state_dict(sd)
print("SRDenseNet loaded successfully")
