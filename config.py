import os
from tools import create_dir
from data_io import get_texture_iter

# Create necessary output directories
create_dir('samples')  # create the directory if necessary for the output samples 
create_dir('models') 

home = os.path.expanduser("~")

def zx_to_npx(zx, depth):
    """
    Calculates the size of the output image given a stack of 'same' padded
    convolutional layers with size depth, and the size of the input field zx.
    The formula reflects the output size after passing through depth layers.
    """
    # Each layer roughly doubles the spatial size
    return (zx - 1) * 2**depth + 1

class Config:
    """
    Wraps all configuration parameters in 'static' variables.
    These parameters define optimization, network structure, and sampling settings.
    """
    
    # Optimization constants
    lr = 0.0002  # learning rate of Adam optimizer
    b1 = 0.5  # beta1 parameter for Adam (momentum term)
    l2_fac = 1e-8  # L2 weight regularization factor
    epoch_count = 100  # Number of epochs to train
    k = 1  # Number of D updates for each G update
    batch_size = 25
    epoch_iters = batch_size * 1000  # Number of steps inside one epoch 
                 
    def __init__(self):    
        # Sampling parameters    
        self.nz_local = 30  # Number of local Z dimensions    
        self.nz_global = 60  # Number of global Z dimensions
        self.nz_periodic = 3  # Number of periodic Z dimensions
        self.nz_periodic_MLPnodes = 50  # Number of MLP nodes for periodic gating
        self.nz = self.nz_local + self.nz_global + self.nz_periodic * 2  # Total Z dimensions
        self.periodic_affine = False  # If True, uses planar waves sum of x, y sinusoids
        self.zx = 6  # Number of spatial dimensions in Z
        self.zx_sample = 32  # Spatial dimension size for producing the samples    
        self.zx_sample_quilt = self.zx_sample // 4  # Number of tiles in the global dimension quilt for output sampling

        # Network parameters
        self.nc = 3  # Number of channels in input X (i.e., r, g, b)
        self.gen_ks = ([(5, 5)] * 5)[::-1]  # Kernel sizes on each layer - should be odd numbers
        self.dis_ks = [(5, 5)] * 5  # Kernel sizes for the discriminator layers
        self.gen_ls = len(self.gen_ks)  # Number of layers in the generator
        self.dis_ls = len(self.dis_ks)  # Number of layers in the discriminator
        self.gen_fn = [self.nc] + [2**(n + 6) for n in range(self.gen_ls - 1)]  # Generator filters
        self.gen_fn = self.gen_fn[::-1]  # Reverse order for the generator filters
        self.dis_fn = [2**(n + 6) for n in range(self.dis_ls - 1)] + [1]  # Discriminator filters
        self.npx = zx_to_npx(self.zx, self.gen_ls)  # Output image size after generator layers
        
        # Input texture folder
        self.sub_name = "honey"  # Subfolder name for textures
        self.texture_dir = os.path.join(home, "DILOG", "dcgan_code-master", "texture_gan", f"{self.sub_name}")
        # Ensure the texture directory exists
        if not os.path.exists(self.texture_dir):
            raise FileNotFoundError(f"Texture directory '{self.texture_dir}' does not exist.")

        # Name of the file for saving models and samples
        self.save_name = f"{self.sub_name}_filters{self.dis_fn[0]}_npx{self.npx}_{self.gen_ls}gL_{self.dis_ls}dL_{self.nz_global}Global_{self.nz_periodic}Periodic_{self.periodic_affine}Affine_{self.nz_local}Local"
        self.load_name = None  # Set to None for initializing network from scratch
           
    def data_iter(self):
        """
        Returns the correct data iterator based on class variables.
        This method avoids the issue of trying to pickle iterator objects in Python.
        """
        return get_texture_iter(self.texture_dir, npx=self.npx, mirror=False, batch_size=self.batch_size)

    def print_info(self):
        """
        Outputs information about the current configuration and setup.
        """
        print(f"Learning and generating samples from zx {self.zx}, which yields images of size npx {zx_to_npx(self.zx, self.gen_ls)}")
        print(f"Producing samples from zx_sample {self.zx_sample}, which yields images of size npx {zx_to_npx(self.zx_sample, self.gen_ls)}")
        print(f"Saving samples and model data to file {self.save_name}")