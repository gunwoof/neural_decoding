import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib

### file 경로 ###
SUB = "sub-01"
base_dir = "/research/wsi/research/03-Neural_decoding/3-bids"
mni_dir = os.path.join(base_dir, "1-beta", "beta_mni", SUB)

train_wb_z = os.path.join(mni_dir, f"{SUB}_beta_train_wb_z.npy")
test_wb_z = os.path.join(mni_dir, f"{SUB}_beta_test_wb_z.npy")

train_tsv = os.path.join(mni_dir, f"{SUB}_beta_train.tsv")
test_tsv = os.path.join(mni_dir, f"{SUB}_beta_test.tsv")

mask = os.path.join(base_dir, "1-beta", "beta_mni", "schaefer_mask.nii.gz")

# atlas
schaefer_100 = os.path.join(base_dir,"2-derivatives","Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")
schaefer_200 = os.path.join(base_dir,"2-derivatives","Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")
schaefer_400 = os.path.join(base_dir,"2-derivatives","Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")
