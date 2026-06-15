import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block."""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RealESRGAN_RRDBNet(nn.Module):
    """
    RRDBNet architecture customized for RealESRGAN.
    This exactly matches the key structure expected by the RealESRGAN_x4plus weights.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RealESRGAN_RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Real-ESRGAN uses 'body' for the RRDB trunk
        self.body = nn.Sequential(*[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RealESRGANModel(nn.Module):
    """
    Wrapper model that handles image processing for Real-ESRGAN gracefully:
    - Reflection padding to avoid edge artifacts
    - Input tensor normalization safety
    - Automatically reduces RGB output to Grayscale if intended.
    """
    def __init__(self, weight_path=None, pre_pad=10, half=False):
        super(RealESRGANModel, self).__init__()
        self.pre_pad = pre_pad
        self.scale = 4
        
        self.model = RealESRGAN_RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale
        )
        
        if weight_path:
            self.load_weights(weight_path)
            
        if half:
            self.model = self.model.half()
            
    def load_weights(self, weight_path):
        sd = torch.load(weight_path, map_location='cpu', weights_only=False)
        if 'params_ema' in sd: sd = sd['params_ema']
        elif 'ema_state_dict' in sd: sd = sd['ema_state_dict']
        elif 'state_dict' in sd: sd = sd['state_dict']
        elif 'params' in sd: sd = sd['params']

        # Strip standard prefixes
        sd = {k.replace('module.', '').replace('model.', ''): v for k, v in sd.items()}

        # For RealESRGAN_x4-DFO2K, keys might use 'RRDB_trunk', 'trunk_conv'
        new_sd = {}
        for k, v in sd.items():
            new_k = k
            if 'RRDB_trunk' in new_k:
                new_k = new_k.replace('RRDB_trunk.', 'body.')
                new_k = new_k.replace('.RDB', '.rdb')
            new_k = new_k.replace('trunk_conv.', 'conv_body.')
            new_k = new_k.replace('upconv1.', 'conv_up1.')
            new_k = new_k.replace('upconv2.', 'conv_up2.')
            new_k = new_k.replace('HRconv.', 'conv_hr.')
            new_sd[new_k] = v
            
        self.model.load_state_dict(new_sd, strict=True)
            
    def forward(self, img_tensor):
        """
        img_tensor: Tensor of shape (B, C, H, W) in range [0, 1]
        Can be (B, 1, H, W) grayscale or (B, 3, H, W) RGB.
        """
        B, C, H, W = img_tensor.shape
        
        # Real-ESRGAN expects 3-channel
        if C == 1:
            inp = img_tensor.repeat(1, 3, 1, 1)
        else:
            inp = img_tensor
            
        # Pad slightly to avoid border artifacts
        if self.pre_pad > 0:
            inp = F.pad(inp, (self.pre_pad, self.pre_pad, self.pre_pad, self.pre_pad), 'reflect')
            
        # Generate output
        out = self.model(inp)
        
        # Remove padding
        if self.pre_pad > 0:
            scale = self.scale
            pad_scaled = self.pre_pad * scale
            out = out[:, :, pad_scaled : -pad_scaled, pad_scaled : -pad_scaled]
            
        # Optional: mitigate pure black artifact by clamping small values
        out = out.clamp_(0, 1)
        
        # If input was 1-channel, ensure output is 1-channel (average RGB)
        if C == 1:
            out = out.mean(dim=1, keepdim=True)
            
        return out
