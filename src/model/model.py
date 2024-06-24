from .layers import *

class FourierDiffusionModel(nn.Module):
    def __init__(self, timesteps):
        super(FourierDiffusionModel, self).__init__()
        self.fourier_transform = FourierTransformLayer()
        self.inverse_fourier_transform = InverseFourierTransformLayer()
        self.high_freq_conv = HighFrequencyConvLayer()
        self.low_freq_conv = LowFrequencyConvLayer()
        self.forward_process = ForwardProcessLayer(timesteps)
        self.reverse_process = ReverseProcessLayer(timesteps)

    def forward(self, x, t):
        # Fourier Transform
        x_ft = self.fourier_transform(x)
        
        # Split high and low frequencies
        x_ft_high, x_ft_low = torch.chunk(x_ft, 2, dim=1)
        
        # Process high frequencies
        x_ft_high = self.high_freq_conv(x_ft_high)
        x_ft_high, noise = self.forward_process(x_ft_high, t)
        x_ft_high = self.reverse_process(x_ft_high, t)
        
        # Process low frequencies
        x_ft_low = self.low_freq_conv(x_ft_low)
        x_ft_low, noise = self.forward_process(x_ft_low, t)
        x_ft_low = self.reverse_process(x_ft_low, t)
        
        # Combine high and low frequencies
        x_ft_combined = torch.cat((x_ft_high, x_ft_low), dim=1)
        
        # Inverse Fourier Transform
        x_reconstructed = self.inverse_fourier_transform(x_ft_combined)
        
        return x_reconstructed
