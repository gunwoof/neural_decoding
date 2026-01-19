'''
session별 whole brain beta .nii 파일 -> session별 whole brain beta npy 파일로 변환 및 저장

input: sub-*_ses-*_beta_wb.nii.gz / schaefer_mni_mask.nii.gz
output: sub-*_ses-*_beta_wb.npy

'''

import os
import glob
import nibabel as nib
import numpy as np

SUB = "sub-05"
base_dir = f"/research/wsi/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni"
beta_dir = f"{base_dir}/{SUB}"
mask_path = f"{base_dir}/schaefer_mni_mask.nii.gz"

# 마스크 불러와 flatten
mask_img = nib.load(mask_path)
mask = mask_img.get_fdata(dtype=np.float32)
mask_bool = (mask > 0.5)
mask_flat = mask_bool.reshape(-1)
n_vox = int(mask_flat.sum())
print(f"Mask shape: {mask.shape}, n_voxels in mask: {n_vox}")

# 4d beta 파일 불러오기
beta_pattern = os.path.join(beta_dir, f"{SUB}_ses-*_beta_mni.nii.gz")
beta_paths = sorted(glob.glob(beta_pattern))
print(f"Found {len(beta_paths)} beta files.")


# beta 파일 마스킹 및 npy로 저장
for beta_path in beta_paths:
    fname = os.path.basename(beta_path)
    print("Processing:", fname)

    beta_img = nib.load(beta_path)
    beta_data = beta_img.get_fdata(dtype=np.float32)    # shape: (182, 218, 182, 750)

    if beta_data.shape[:3] != mask_bool.shape:
        raise ValueError(f"Shape mismatch: beta {beta_data.shape} vs mask {mask_bool.shape}")

    nx, ny, nz, T = beta_data.shape
    beta_flat = beta_data.reshape(-1, T)  # shape : (182*218*182, 750)
    beta_masked = beta_flat[mask_flat] # shape : (101319, 750)

    # 저장 경로
    out_npy = beta_path.replace("_beta_mni.nii.gz", "_beta_wb.npy")
    np.save(out_npy, beta_masked.astype(np.float32))
    print("  -> Saved", out_npy, beta_masked.shape)

print("모든 ses 파일 마스킹 완료!")
