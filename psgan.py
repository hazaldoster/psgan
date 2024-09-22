import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Helper function for initializing weights
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)

class PeriodicLayer(nn.Module):
    def __init__(self, config, wave_params):
        super(PeriodicLayer, self).__init__()
        self.config = config
        self.wave_params = wave_params
        self.nPeriodic = config.nz_periodic
        
    def forward(self, Z):
        if self.nPeriodic == 0:
            return Z
        
        nPeriodic = self.nPeriodic
        band0 = self.wave_params[0].unsqueeze(0).unsqueeze(2).unsqueeze(3)  # Broadcast to batch size
        if self.config.periodic_affine:
            band1 = Z[:, -nPeriodic*2::2] * band0[:, :nPeriodic] + Z[:, -nPeriodic*2+1::2] * band0[:, nPeriodic:2*nPeriodic]
            band2 = Z[:, -nPeriodic*2::2] * band0[:, 2*nPeriodic:3*nPeriodic] + Z[:, -nPeriodic*2+1::2] * band0[:, 3*nPeriodic:]
        else:
            band1 = Z[:, -nPeriodic*2::2] * band0[:, :nPeriodic]
            band2 = Z[:, -nPeriodic*2+1::2] * band0[:, 3*nPeriodic:]
        
        band = torch.cat([band1, band2], dim=1)
        return torch.cat([Z[:, :-2*nPeriodic], torch.sin(band)], dim=1)

# Generator Network
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.nz = config.nz
        
        # Define layers
        self.layers = nn.ModuleList()
        self.layers.append(PeriodicLayer(config, self._setup_wave_params()))
        
        for i in range(len(config.gen_fn) - 1):
            self.layers.append(nn.ConvTranspose2d(config.gen_fn[i], config.gen_fn[i + 1], kernel_size=config.gen_ks[i], stride=2, padding=1))
            self.layers.append(nn.BatchNorm2d(config.gen_fn[i + 1]))
            self.layers.append(nn.ReLU(True))
        
        self.final = nn.ConvTranspose2d(config.gen_fn[-1], config.nc, kernel_size=config.gen_ks[-1], stride=2, padding=1)
        self.final_activation = nn.Tanh()

        # Initialize weights
        self.apply(weights_init_normal)
        
    def _setup_wave_params(self):
        """ Set up the parameters of the periodic dimensions """
        if self.config.nz_periodic:
            nPeriodic = self.config.nz_periodic
            wave_params = torch.randn(nPeriodic * 2 * 2, requires_grad=True)
            return wave_params
        return None

    def forward(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return self.final_activation(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.layers = nn.ModuleList()
        
        # Define layers
        self.layers.append(nn.Conv2d(config.nc, config.dis_fn[0], kernel_size=config.dis_ks[0], stride=2, padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        for i in range(1, len(config.dis_fn)):
            self.layers.append(nn.Conv2d(config.dis_fn[i-1], config.dis_fn[i], kernel_size=config.dis_ks[i], stride=2, padding=1))
            self.layers.append(nn.BatchNorm2d(config.dis_fn[i]))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.final = nn.Conv2d(config.dis_fn[-1], 1, kernel_size=4, stride=1, padding=0)
        self.final_activation = nn.Sigmoid()

        # Initialize weights
        self.apply(weights_init_normal)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return self.final_activation(x).view(-1, 1).squeeze(1)

# Noise sampling function (replacing Theano's random tensor sampling)
def sample_noise_tensor(config, batch_size, zx, zx_quilt=None):
    Z = torch.zeros((batch_size, config.nz, zx, zx))
    Z[:, config.nz_global:config.nz_global+config.nz_local] = torch.rand((batch_size, config.nz_local, zx, zx)) * 2 - 1

    if zx_quilt is None:
        Z[:, :config.nz_global] = torch.rand((batch_size, config.nz_global, 1, 1)) * 2 - 1
    else:
        for i in range(zx // zx_quilt):
            for j in range(zx // zx_quilt):
                Z[:, :config.nz_global, i*zx_quilt:(i+1)*zx_quilt, j*zx_quilt:(j+1)*zx_quilt] = \
                    torch.rand((batch_size, config.nz_global, 1, 1)) * 2 - 1

    if config.nz_periodic > 0:
        for i, pixel in enumerate(torch.linspace(30, 130, config.nz_periodic)):
            band = np.pi * (0.5 * (i + 1) / config.nz_periodic + 0.5)
            for h in range(zx):
                Z[:, -2 * (i + 1), :, h] = h * band
            for w in range(zx):
                Z[:, -2 * (i + 1) + 1, w] = w * band

    return Z
