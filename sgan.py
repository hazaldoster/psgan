#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import joblib

from config import Config
from tools import TimePrint
from data_io import get_texture_iter, save_tensor


# Helper function to initialize weights
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.layers = nn.ModuleList()

        # Build layers based on the config
        for i in range(len(config.gen_fn) - 1):
            self.layers.append(nn.ConvTranspose2d(config.gen_fn[i], config.gen_fn[i+1], kernel_size=config.gen_ks[i], stride=2, padding=1))
            self.layers.append(nn.BatchNorm2d(config.gen_fn[i+1]))
            self.layers.append(nn.ReLU(True))
        
        self.final = nn.ConvTranspose2d(config.gen_fn[-1], config.nc, kernel_size=config.gen_ks[-1], stride=2, padding=1)
        self.final_activation = nn.Tanh()

        # Initialize weights
        self.apply(weights_init_normal)

    def forward(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return self.final_activation(x)

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.layers = nn.ModuleList()

        # First layer without batch normalization
        self.layers.append(nn.Conv2d(config.nc, config.dis_fn[0], kernel_size=config.dis_ks[0], stride=2, padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Build other layers with batch normalization
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

# Sample noise for the generator
def sample_noise_tensor(config, batch_size, zx):
    return torch.randn(batch_size, config.nz, zx, zx)

# SGAN class
class SGAN:
    def __init__(self, name=None):
        self.config = Config()
        if name is not None:
            print(f"Loading parameters from file: {name}")
            vals = joblib.load(name)

            # Load parameters into models
            self.generator = Generator(self.config)
            self.discriminator = Discriminator(self.config)
            self.generator.load_state_dict(vals['gen_state'])
            self.discriminator.load_state_dict(vals['dis_state'])
        else:
            # Initialize new models
            self.generator = Generator(self.config)
            self.discriminator = Discriminator(self.config)

        # Loss and Optimizers
        self.criterion = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.config.lr, betas=(self.config.b1, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.config.lr, betas=(self.config.b1, 0.999))

    def save(self, name):
        print(f"Saving SGAN parameters in file: {name}")
        vals = {
            "config": self.config,
            "gen_state": self.generator.state_dict(),
            "dis_state": self.discriminator.state_dict()
        }
        joblib.dump(vals, name)

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            Gcost, Dcost = [], []
            iters = len(data_loader)

            for i, real_samples in enumerate(tqdm(data_loader)):
                real_labels = torch.ones(real_samples.size(0), 1)
                fake_labels = torch.zeros(real_samples.size(0), 1)

                # Train Discriminator
                self.optimizer_d.zero_grad()

                # Real data loss
                output_real = self.discriminator(real_samples)
                loss_real = self.criterion(output_real, real_labels)

                # Fake data loss
                noise = sample_noise_tensor(self.config, real_samples.size(0), self.config.zx)
                fake_samples = self.generator(noise)
                output_fake = self.discriminator(fake_samples.detach())
                loss_fake = self.criterion(output_fake, fake_labels)

                d_loss = loss_real + loss_fake
                d_loss.backward()
                self.optimizer_d.step()

                Dcost.append(d_loss.item())

                # Train Generator
                self.optimizer_g.zero_grad()

                output_fake = self.discriminator(fake_samples)
                g_loss = self.criterion(output_fake, real_labels)
                g_loss.backward()
                self.optimizer_g.step()

                Gcost.append(g_loss.item())

            print(f"Epoch {epoch + 1}/{num_epochs} | Gcost = {np.mean(Gcost):.4f}, Dcost = {np.mean(Dcost):.4f}")

            # Save images and model checkpoint
            with torch.no_grad():
                sample_noise = sample_noise_tensor(self.config, 1, self.config.zx_sample)
                generated_sample = self.generator(sample_noise).cpu()
                save_tensor(generated_sample[0], f'samples/{self.config.save_name}_epoch{epoch}.jpg')
                self.save(f'models/{self.config.save_name}_epoch{epoch}.sgan')

if __name__ == "__main__":
    c = Config()

    # Load SGAN model or create a new one
    if c.load_name is None:
        sgan = SGAN()
    else:
        sgan = SGAN(name=os.path.join('models', c.load_name))

    c.print_info()

    # Generate random sample for testing
    z_sample = torch.randn(1, c.nz, c.zx_sample, c.zx_sample)

    # Simulate DataLoader (this should be replaced by actual data loading logic)
    data_loader = get_texture_iter(c.batch_size)

    # Train SGAN
    sgan.train(data_loader, c.epoch_count)