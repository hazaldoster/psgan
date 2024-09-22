import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import random

def image_to_tensor(img):
    """
    Convert image to PyTorch tensor format;
    changes channel dimension to be in the first position and rescales from [0, 255] to [-1, 1].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),             # Converts to range [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Rescale to [-1, 1]
    ])
    return transform(img)

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor (3-channel) to a PIL image;
    changes channel order from [C, H, W] to [H, W, C] and rescales from [-1, 1] to [0, 255].
    """
    tensor = (tensor * 0.5) + 0.5  # Rescale to [0, 1]
    tensor = tensor.clamp(0, 1)  # Ensure all values are in the valid range
    img = transforms.ToPILImage()(tensor)
    return img

def get_texture_iter(folder, npx=128, batch_size=64, filter=None, mirror=True):
    """
    Iterate over image patches extracted from images in a folder.
    
    @param folder: Path to the folder containing images.
    @param npx: Size of patches to extract.
    @param batch_size: Number of images in each batch.
    @param filter: (Unused parameter; can be removed).
    @param mirror: If True, augment the dataset by mirroring images left-right.
    @return: A generator yielding batches of image patches of size (batch_size, 3, npx, npx), with values in [-1, 1].
    """
    HW = npx
    imTex = []
    files = os.listdir(folder)
    for f in files:
        name = os.path.join(folder, f)
        try:
            img = Image.open(name).convert('RGB')  # Ensure image has 3 channels (RGB)
            imTex.append(image_to_tensor(img))
            if mirror:
                img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
                imTex.append(image_to_tensor(img_mirror))
        except Exception as e:
            print(f"Image {name} failed to load! Error: {e}")

    while True:
        data = torch.zeros((batch_size, 3, npx, npx))  # PyTorch tensor of zeros, assumes 3 channels
        for i in range(batch_size):
            ir = random.randint(0, len(imTex) - 1)
            imgBig = imTex[ir]
            # Sample patches if the patch size is smaller than the image dimensions
            if HW < imgBig.shape[1] and HW < imgBig.shape[2]:
                h = random.randint(0, imgBig.shape[1] - HW)
                w = random.randint(0, imgBig.shape[2] - HW)
                img = imgBig[:, h:h + HW, w:w + HW]
            else:  # If the image is smaller, use the whole image
                img = imgBig
            data[i] = img
        yield data

def save_tensor(tensor, filename):
    """
    Save a PyTorch tensor (3-channel, [C, H, W]) to an image file.
    """
    img = tensor_to_image(tensor)
    img.save(filename)

if __name__ == "__main__":
    print("nothing here.")