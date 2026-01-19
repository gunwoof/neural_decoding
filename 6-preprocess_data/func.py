import numpy as np
import nibabel as nib

def print_npy_shape(path, mmap_mode=None):
    x = np.load(path, mmap_mode=mmap_mode)
    print(f"{path}: type={type(x)}, shape={getattr(x, 'shape', None)}, dtype={getattr(x, 'dtype', None)}")

def print_npz_shape(path):
    data = np.load(path)
    for key in data.files:
        x = data[key]
        print(f"  {key}: type={type(x)}, shape={getattr(x, 'shape', None)}, dtype={getattr(x, 'dtype', None)}")

def print_nii_metadata(path):
    img = nib.load(path)
    print(f"{path}: shape={img.shape}, affine=\n{img.affine}, header=\n{img.header}")