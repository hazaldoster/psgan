import torch
import sys
import numpy as np
from data_io import save_tensor
from psgan import PSGAN, sample_noise_tensor  # Assuming PSGAN has been converted to PyTorch

if len(sys.argv) <= 1:
    print("Please give model filename")
    print("e.g. checked github model hex1_filters64_npx161_5gL_5dL_0Global_3Periodic_15Local_epoch43.sgan")
    raise Exception('No filename specified')

name = sys.argv[1]
print(f"Using stored model: {name}")
print(torch.__version__)
def mosaic_tile(psgan, NZ1=12, NZ2=12, repeat=(2, 3)):
    ovp = 2  # Overlap for tiling
    tot_subsample = 2 ** psgan.gen_depth  # Total subsampling from generator depth
    print(f"NZ1 NZ2 for tilable texture: {NZ1}, {NZ2}")

    # Sample the noise tensor for tiling
    sample_zmb = sample_noise_tensor(psgan.config, 1, max(NZ1, NZ2))[:, :, :NZ1, :NZ2]

    # Apply overlap
    sample_zmb[:, :, :, -ovp * 2:] = sample_zmb[:, :, :, :ovp * 2]
    sample_zmb[:, :, -ovp * 2:, :] = sample_zmb[:, :, :ovp * 2, :]

    # Generate the samples using the generator
    samples = psgan.generate(sample_zmb)

    # Offset loss calculation to find the best tile overlap
    def offsetLoss(crop1, crop2):
        return torch.abs(samples[:, :, :, crop1] - samples[:, :, :, -crop2]).mean() + \
               torch.abs(samples[:, :, crop1] - samples[:, :, -crop2]).mean()

    best = float('inf')
    crop1, crop2 = 0, 0
    for i in range(ovp * tot_subsample // 2, ovp * tot_subsample):
        for j in range(ovp * tot_subsample // 2, ovp * tot_subsample):
            loss = offsetLoss(i, j)
            if loss < best:
                best = loss
                crop1 = i
                crop2 = j

    print(f"Optimal offsets: {crop1}, {crop2}, offset edge errors: {best}")
    
    # Crop the generated samples based on the optimal offsets
    samples = samples[:, :, crop1:-crop2, crop1:-crop2]
    s = (samples.shape[2], samples.shape[3])
    print(f"Tile sample size: {samples.shape}")

    # Save the tile sample
    save_tensor(samples[0].cpu().numpy(), f"samples/TILE_{name.replace('/', '_')}_{s}.jpg")

    # Create a repeated mosaic of the sample if repeat is specified
    if repeat is not None:
        sbig = np.zeros((3, repeat[0] * s[0], repeat[1] * s[1]))
        for i in range(repeat[0]):
            for j in range(repeat[1]):
                sbig[:, i * s[0]:(i + 1) * s[0], j * s[1]:(j + 1) * s[1]] = samples[0].cpu().numpy()
        save_tensor(sbig, f"samples/TILE_{name.replace('/', '_')}_{s}_{repeat}.jpg")

def sample_texture(psgan, NZ1=60, quilt_tile=20):
    z_sample = sample_noise_tensor(psgan.config, 1, NZ1, quilt_tile)
    data = psgan.generate(z_sample)
    save_tensor(data[0].cpu().numpy(), f'samples/stored_{name.replace("/", "_")}.jpg')

# Load the PSGAN model
psgan = PSGAN(name=name)
c = psgan.config
print(f"nz: {c.nz}, global Dimensions: {c.nz_global}, periodic Dimensions: {c.nz_periodic}")
print(f"G values: {c.gen_fn}, {c.gen_ks}")
print(f"D values: {c.dis_fn}, {c.dis_ks}")

# Sample texture and tile mosaic
sample_texture(psgan)
mosaic_tile(psgan)