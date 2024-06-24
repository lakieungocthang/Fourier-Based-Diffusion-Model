import torch
import torch.nn as nn
import torch.fft
from .UNet import UNet

class FourierTransformLayer(nn.Module):
    def __init__(self):
        super(FourierTransformLayer, self).__init__()

    def forward(self, x):
        # Perform Fourier Transform
        x_ft = torch.fft.fftn(x, dim=(-2, -1))
        return x_ft

class InverseFourierTransformLayer(nn.Module):
    def __init__(self):
        super(InverseFourierTransformLayer, self).__init__()

    def forward(self, x_ft):
        # Perform Inverse Fourier Transform
        x = torch.fft.ifftn(x_ft, dim=(-2, -1))
        return x.real

class HighFrequencyConvLayer(nn.Module):
    def __init__(self):
        super(HighFrequencyConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class LowFrequencyConvLayer(nn.Module):
    def __init__(self):
        super(LowFrequencyConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class ForwardProcessLayer(nn.Module):
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02):
        super(ForwardProcessLayer, self).__init__()
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)

    def forward(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise

class ReverseProcessLayer(nn.Module):
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02):
        super(ReverseProcessLayer, self).__init__()
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.model = Unet(in_channels=2, out_channels=1)  # Adjust in_channels according to the data

    def forward(self, x, t):
        pred_noise = self.model(x)
        beta_t = self.betas[t]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_recip_alphas_t * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)