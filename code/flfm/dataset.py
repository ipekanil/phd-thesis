import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageSequence
import numpy as np
from config import config  
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time 
"""

class MyDataset(Dataset):
    def __init__(self,  stack_dir, transform_2d, transform_3d):
        self.stack_dir = stack_dir
        self.files = sorted(os.listdir(stack_dir))
        self.transform_2d = transform_2d
        self.transform_3d = transform_3d

    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):
        start_time = time.time()

        file_name = self.files[idx]
    
        # Construct the paths for Stack using the same filename
        stack_path = os.path.join(self.stack_dir, file_name)

        # Load the images
        t0 = time.time()
        stack_image = self.load_tiff_stack(stack_path)  # Load Stack image D x H x W
        t1 = time.time()
        print(f"⏱️ Time to load TIFF: {t1 - t0:.2f}s")

        print("Data image shape:", stack_image.shape)
        stack_image = stack_image.numpy() 

        # Apply 3D transforms if any
        if self.transform_3d:
            t4 = time.time()
            stack_image = self.transform_3d(stack_image)
            t5 = time.time()
            print(f"🎨 Transform 3D took: {t5 - t4:.2f}s")

        # Print the filenames used
        print("Filenames:")
        print(f"  Stack: {stack_path}")

        #self.visualize_stack(stack_image)

        total = time.time() - start_time
        print(f"✅ __getitem__ total time for '{file_name}': {total:.2f}s\n")

        return stack_image


    def load_tiff_stack(self, path):
        img = Image.open(path)
        frames = [torch.tensor(np.array(frame.convert('L'))) for frame in ImageSequence.Iterator(img)]
        stack_tensor = torch.stack(frames)
        return stack_tensor

    def center_crop(self, img, crop_size):
        # Center crop an image to a specified size 
        h, w = img.shape[-2:]  # Assume shape is (C, H, W)
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        return img[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
    
    def visualize_stack(self, stack_images):
        D, H, W = stack_images.shape  # Get dimensions
        rows = 3
        cols = int(np.ceil(D / rows))  # Calculate columns to fit all slices

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

        for i, ax in enumerate(axes.flat):
            if i < D:
                ax.imshow(stack_images[i], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"Slice {i+1}")
            ax.axis("off")  # Remove axis labels

        plt.tight_layout()
        plt.show()

    


# Testing the dataset
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from torchvision import transforms
    from config import config

    stack_dir = r"D:\F-VCD_FLFM_meta_dataset\Stackx1\train"

    # Define any transformations you want (if none, you can set transform_2d and transform_3d to None)
    transform_2d = config.transform_2d
    transform_3d = config.transform_3d

    # Initialize the dataset
    dataset = MyDataset(stack_dir=stack_dir, transform_2d=transform_2d, transform_3d=transform_3d)

    # Access a single sample from the dataset
    sample_idx = 3  # Change this to test different samples
    stack_images = dataset[sample_idx]

       
    # Print the shapes to check
    print(f"Stack Image Shape: {stack_images.shape}")

    print(f"Stack Image Max: {stack_images.max()}")
    print(f"Stack Image Min: {stack_images.min()}")
    
        
    # The stack images (3D volume)
    # Assuming stack_images is a 3D stack with dimensions [n_slices, height, width]
    # Function to visualize 3D volume
    from mayavi import mlab

    # Assuming stack_images is a PyTorch tensor
    # First, make sure the tensor is on the CPU and convert it to a NumPy array
    if stack_images.is_cuda:
        stack_images = stack_images.cpu()  # Move tensor to CPU if it's on GPU

    # Convert to NumPy
    stack_images_np = stack_images.numpy()

    # Function to visualize 3D volume data
    def visualize_3d_volume(data, title):
        #Visualize a 3D volume data with Mayavi
        mlab.figure(figure=title, bgcolor=(0, 0, 0))
        mlab.contour3d(data, contours=4, transparent=True, colormap='viridis')
        mlab.show()

    
    # Visualize the stack image
    dataset.visualize_stack(stack_images)

    #visualize_3d_volume(stack_images_np, f"Visualization of Stack")

    
 

"""


import os
import time
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
#from mayavi import mlab

class MyDataset(Dataset):
    def __init__(self, stack_dir, config, transform_3d=None):
        self.stack_dir = Path(stack_dir)
        self.files = sorted([p for p in self.stack_dir.iterdir() if p.suffix.lower() in [".h5", ".hdf5"]])
        self.transform_3d = transform_3d

        op = config.optics_params
        self.volume_depth_min = op["volume_depth_min"]
        self.volume_depth_max = op["volume_depth_max"]
        self.depth_resolution = op["depth_resolution"]

        D = (-self.volume_depth_min + self.volume_depth_max) / self.depth_resolution
        self.D = int(np.ceil(D))  # number of Z planes to keep

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        start_time = time.time()
        path = self.files[idx]

        t0 = time.time()
        stack = self.load_h5_volume(path)  # torch tensor Z Y X
        t1 = time.time()
        print(f"⏱️ Time to load H5: {t1 - t0:.2f}s")

        #Crop axial sub-volume to match simulated depth.
        #Example: 44 µm target depth / 0.7 µm per slice ≈ 63 Z planes (centered crop).
        stack = self.center_crop_z(stack, self.D)
        print("Data image shape after cropping:", stack.shape)

        stack = stack.float()

        if self.transform_3d:
            t2 = time.time()
            stack_norm, lo, hi = self.transform_3d(stack)   # return_params must be True
            t3 = time.time()
            print(f"🎨 Transform 3D took: {t3 - t2:.2f}s")
            return stack_norm, lo, hi

        total = time.time() - start_time
        print(f"✅ __getitem__ total time for '{path.name}': {total:.2f}s\n")

        return stack, None, None


    def load_h5_volume(self, path: Path):
        with h5py.File(str(path), "r") as hf:
            vol = hf["volume"][...]  # numpy Z Y X

        # Convert to torch float32
        vol = torch.from_numpy(vol)

        # If stored as uint16, keep as float for training
        if vol.dtype != torch.float32:
            vol = vol.float()

        return vol

    def center_crop_z(self, vol_zyx: torch.Tensor, D_keep: int):
        z = vol_zyx.shape[0]
        if D_keep > z:

            raise ValueError(f"Requested D={D_keep} but volume has only Z={z} slices: {z}")

        z0 = (z - D_keep) // 2
        z1 = z0 + D_keep
        return vol_zyx[z0:z1]    

if __name__ == "__main__":
    from config import config
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    def view_volume_slices(vol_zyx, vmin=None, vmax=None):
        if torch.is_tensor(vol_zyx):
            vol = vol_zyx.detach().cpu().numpy()
        else:
            vol = np.asarray(vol_zyx)

        zmax = vol.shape[0] - 1
        z = zmax // 2

        fig, ax = plt.subplots()
        im = ax.imshow(vol[z], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"z = {z}/{zmax}  use scroll wheel")
        ax.axis("off")

        def on_scroll(event):
            nonlocal z
            if event.button == "up":
                z = min(z + 1, zmax)
            else:
                z = max(z - 1, 0)
            im.set_data(vol[z])
            ax.set_title(f"z = {z}/{zmax}  use scroll wheel")
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("scroll_event", on_scroll)
        plt.show()

    def show_mips(vol_zyx):
        if torch.is_tensor(vol_zyx):
            v = vol_zyx.detach().cpu()
        else:
            v = torch.from_numpy(np.asarray(vol_zyx))

        mip_xy = v.max(dim=0).values
        mip_xz = v.max(dim=1).values
        mip_yz = v.max(dim=2).values

        plt.figure()
        plt.imshow(mip_xy, cmap="gray")
        plt.title("MIP XY")
        plt.axis("off")
        plt.savefig("mip_xy.png")
        plt.close()

        plt.figure()
        plt.imshow(mip_xz, cmap="gray")
        plt.title("MIP XZ")
        plt.axis("off")
        plt.savefig("mip_xz.png")
        plt.close()

        plt.figure()
        plt.imshow(mip_yz, cmap="gray")
        plt.title("MIP YZ")
        plt.axis("off")
        plt.savefig("mip_yz.png")
        plt.close()


    #stack_dir = r"E:\confocal_data\cropped_tiles_256_357_ch0ch1_h5"
    stack_dir = r"E:\FLFM_dataset\train" 
    dataset = MyDataset(stack_dir=stack_dir, config=config,  transform_3d=config.transform_3d)

    sample_idx = 1 #1000 # Change this to test different samples
    stack = dataset[sample_idx]
    print("Stack shape:", stack.shape)
    print("Max:", stack.max().item(), "Min:", stack.min().item())

    
    view_volume_slices(stack)
    show_mips(stack)
