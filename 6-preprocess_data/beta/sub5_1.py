'''
session별 whole brain beta npy 파일 -> train/test 분리 및 저장

input: sub-*_ses-*_beta_wb.npy / nsd_stim_info_merged.csv, responses.tsv
output: sub-*_beta_train_wb.npy, sub-*_beta_test_wb.npy

'''

# import libraries
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib

### 경로 설정 ###
SUB = "sub-05"
base_dir = "/research/wsi/research/03-Neural_decoding/3-bids/2-derivatives/1-beta"

mni_dir = os.path.join(base_dir, "beta_mni", SUB)
nsd_stim_info_path = os.path.join(base_dir, "beta_nsd", "nsd_stim_info_merged.csv")
responses_tsv_path = os.path.join(base_dir, "beta_nsd", SUB, "responses.tsv")

beta_train_path = os.path.join(mni_dir, f"{SUB}_beta_train_wb.npy")
beta_test_path = os.path.join(mni_dir, f"{SUB}_beta_test_wb.npy")

### beta 파일들 순서대로 로드  ###
pattern = os.path.join(mni_dir, f"{SUB}_ses-*_beta_wb.npy")
beta_files = sorted(glob.glob(pattern))

print(f"Found {len(beta_files)} beta files:")

beta_list = []
for file in beta_files:
    beta_list.append(np.load(file))
    print(os.path.basename(file), "shape:", beta_list[-1].shape)

beta_all = np.concatenate(beta_list, axis=1)    # (101319, 750) * 40 = (101319, 30000)
print("Concatenated beta shape:", beta_all.shape)

resp = pd.read_csv(responses_tsv_path, sep="\t")
kid_seq = resp["73KID"].astype(int).to_numpy()

stim = pd.read_csv(nsd_stim_info_path)
stim_idx = kid_seq - 1
stim = stim.set_index("num")
shared_col = stim.loc[stim_idx, "shared1000"]
shared_flags = shared_col.to_numpy()


print("총 shared1000=True 개수:", shared_flags.sum())
print("총 shared1000=False 개수:", (~shared_flags).sum())

train_mask = shared_flags
test_mask = ~shared_flags

beta_train = beta_all[:, train_mask]
beta_test = beta_all[:, test_mask]

beta_train = beta_train.T
beta_test = beta_test.T

print("Train beta shape:", beta_train.shape)    # (27000, 101319)
print("Test beta shape:", beta_test.shape)      # (3000, 101319)

np.save(beta_train_path, beta_train)
print("Saved train beta to", beta_train_path)
np.save(beta_test_path, beta_test)
print("Saved test beta to", beta_test_path)
