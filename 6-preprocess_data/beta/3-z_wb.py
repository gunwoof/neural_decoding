'''
각 subject의 whole brain beta의 train, test npy 파일을 train의 mean, std로 정규화

input: sub-*_beta_train_wb.npy, sub-*_beta_test_wb.npy
output: sub-*_beta_train_wb_z.npy, sub-*_beta_test_wb_z.npy, sub-*_wb_train.npz, sub-*_wb_test.npz

'''

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib

### 경로 설정 ###
SUB = "sub-02"
base_dir = "/research/wsi/research/03-Neural_decoding/3-bids/2-derivatives"
mni_dir = os.path.join(base_dir, "1-beta", "beta_mni", SUB)

#input
train_wb = os.path.join(mni_dir, f"{SUB}_beta_train_wb.npy")
test_wb  = os.path.join(mni_dir, f"{SUB}_beta_test_wb.npy")

train_tsv = os.path.join(mni_dir, f"{SUB}_beta_train.tsv")
test_tsv = os.path.join(mni_dir, f"{SUB}_beta_test.tsv")

#output
train_wb_z = os.path.join(mni_dir, f"{SUB}_beta_train_wb_z.npy")
test_wb_z = os.path.join(mni_dir, f"{SUB}_beta_test_wb_z.npy")

train_out_npz = os.path.join(mni_dir, f"{SUB}_wb_train.npz")
test_out_npz = os.path.join(mni_dir, f"{SUB}_wb_test.npz")


### WB npy 로드 ###
wb_train = np.load(train_wb)    # shape (27000, 1055685)
wb_test = np.load(test_wb)      # shape (3000, 1055685)

print("WB train shape:", wb_train.shape)
print("WB test shape:", wb_test.shape)

### WB train 기준 정규화 ###
mean_vox = wb_train.mean(axis=0)
std_vox = wb_train.std(axis=0, ddof=0)
eps = 1e-8
std = np.where(std_vox < eps, 1.0, std_vox)
wb_train_z = (wb_train - mean_vox) / std
wb_test_z = (wb_test - mean_vox) / std

np.save(train_wb_z, wb_train_z)
print("Saved z-scored train WB to", train_wb_z)
np.save(test_wb_z, wb_test_z)
print("Saved z-scored test WB to", test_wb_z)

### npz 파일로 저장 ###
np.savez(train_out_npz, beta=wb_train_z, stimuli=pd.read_csv(train_tsv, sep="\t", header=None)[0].values)
print("Saved train npz to", train_out_npz)
np.savez(test_out_npz, beta=wb_test_z, stimuli=pd.read_csv(test_tsv, sep="\t", header=None)[0].values)
print("Saved test npz to", test_out_npz)