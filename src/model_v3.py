import torch
import torch.nn as nn
import torch.nn.functional as F


# Hardwires the exact shape of the jagged obstacle into the AI's attention.

class GeometricRouting(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # A lightweight 1x1 convolution to let the AI guess what's important
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, sdf_map):
        b, c, h, w = features.shape
        
        #Shrink the massive 64x64 SDF down to the 16x16 bottleneck size
        sdf_resized = F.interpolate(sdf_map, size=(h, w), mode='bilinear', align_corners=False)
        
        #Invert the geometry: Empty space gets ignored, boundaries become massive multipliers
        boundary_focus = torch.exp(-torch.abs(sdf_resized) * 5.0) 
        
        #Combine the AI's internal guess with the absolute laws of geometry
        ai_attention = self.sigmoid(self.attention_conv(features))
        
        # Soft Attention: Baseline of 1.0 (so it doesn't go blind) + Boundary Boost
        hybrid_attention = 1.0 + (ai_attention * boundary_focus)
        
        return features * hybrid_attention

# THE TELESCOPE (2D Spectral Fourier Convolution)
# Converts the wind tunnel into global waves to resolve physics instantly.

class SpectralFourier2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes 
        
        # The complex math weights, Real and Imaginary numbers for waves
        self.scale = (1 / (in_channels * out_channels))
        self.weights_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes))
        self.weights_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes))

    def forward(self, x):
        b, c, h, w = x.shape
        
        #Convert spatial pixels into Fourier waves
        x_ft = torch.fft.rfft2(x)
        
        #Create an empty tensor to hold the complex multiplication results
        out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        #Multiply the waves by our learned parameters
        complex_weights = torch.complex(self.weights_real, self.weights_imag)
        
        # (This is a highly optimized Einstein Summation for complex matrix math) :D
        out_ft[:, :, :self.modes, :self.modes] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, :self.modes, :self.modes], 
            complex_weights
        )
        
        #Convert the waves BACK into spatial pixels
        x_out = torch.fft.irfft2(out_ft, s=(h, w))
        
        return x_out

# The Full Brain

class DoubleConv(nn.Module):
    # Standard U-Net feature extractor
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
    def forward(self, x): return self.net(x)

class HybridUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ENCODER
        # Input: [1 channel SDF + 2 channels Velocity (U, V)] = 3 channels that's fucking awesome
        self.enc1 = DoubleConv(3, 32)   # 64x64
        self.pool1 = nn.MaxPool2d(2)    
        self.enc2 = DoubleConv(32, 64)  # 32x32
        self.pool2 = nn.MaxPool2d(2)    
        self.enc3 = DoubleConv(64, 128) # 16x16
        
        
        self.routing_block = GeometricRouting(in_channels=128)
        self.fourier_block = SpectralFourier2D(in_channels=128, out_channels=128, modes=8)
        
        # DECODER (Macro-Vision + Deep Supervision)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64) # 128 
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        
        # THE THREE GRADING OUTPUTS
        # The Deep Macro prediction (16x16)
        self.out_16 = nn.Conv2d(128, 2, kernel_size=1) 
        
        # The Mid-level prediction (32x32)
        self.out_32 = nn.Conv2d(64, 2, kernel_size=1)
        
        # The Final High-Res prediction (64x64)
        self.out_64 = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        # We need the pure SDF map to feed the Routing Block later
        sdf_map = x[:, 0:1, :, :] 
        
        # Encoder 
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        bottleneck_features = self.enc3(self.pool2(e2))
        
        # The Hybrid Bottleneck Magic
        routed_features = self.routing_block(bottleneck_features, sdf_map)
        fourier_features = self.fourier_block(routed_features)
        
        # Deep Supervision Target 1
        pred_16 = self.out_16(fourier_features) 
        
        # Decoder
        d2 = self.up2(fourier_features)
        d2 = torch.cat([d2, e2], dim=1) # Skip connection from Encoder 2
        d2 = self.dec2(d2)
        
        # Deep Supervision Target 2
        pred_32 = self.out_32(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1) # Skip connection from Encoder 1
        d1 = self.dec1(d1)
        
        
        pred_64 = self.out_64(d1)
        
        return pred_16, pred_32, pred_64