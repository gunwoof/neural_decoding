import os
import numpy as np
import pandas as pd
import nibabel as nib

#경로 설정
SUB="sub-07"
base_dir = "/research/wsi/research/03-Neural_decoding/3-bids/2-derivatives"
beta_dir = os.path.join(base_dir, "1-beta", "beta_mni", SUB)
beta_train_path = os.path.join(beta_dir, f"{SUB}_beta_train_z.npy")
beta_test_path = os.path.join(beta_dir, f"{SUB}_beta_test_z.npy")

train_tsv_path = os.path.join(beta_dir, f"{SUB}_beta_train.tsv")
test_tsv_path = os.path.join(beta_dir, f"{SUB}_beta_test.tsv")

nsd_mask = os.path.join(base_dir, "1-beta", "beta_mni", "group_nsdgeneral_in_MNI.nii.gz")
schaefer_path = os.path.join(base_dir, "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")

#load paths
beta_train = np.load(beta_train_path)
beta_test = np.load(beta_test_path)
nsd_mask = nib.load(nsd_mask).get_fdata().flatten()
schaefer = nib.load(schaefer_path).get_fdata().flatten()

valid_idx = np.where(nsd_mask == 1)[0]
schaefer_labels = schaefer[valid_idx].astype(int)
unique_labels = np.unique(schaefer_labels)
print("nsdgeneral 안에서 등장한 라벨들:", unique_labels)
exclude_labels = {0, 12, 36, 37, 38, 58, 78, 110}
labels = [int(v) for v in unique_labels if v not in exclude_labels]

#train
train_schaefer = {}
for label in labels:
    idx = np.where(schaefer_labels == label)[0]
    train_schaefer[int(label)] = beta_train[:, idx]

max_train = max(arr.shape[1] for arr in train_schaefer.values())
print("max_train:", max_train)

padded_train = {}
for label, data in train_schaefer.items():
    pad_width = max_train - data.shape[1]
    if pad_width > 0:
        padded = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
    else:
        padded = data
    padded_train[label] = padded

sorted_labels = sorted(padded_train.keys())
train_data = np.stack([padded_train[label] for label in sorted_labels], axis=0) #shape: (num_labels, num_trials, max_voxels)
train_data = np.transpose(train_data, (1, 0, 2)) #shape: (num_trials, num_labels, max_voxels)
print("Final train_data shape:", train_data.shape)

train_npz_path = os.path.join(beta_dir, f"{SUB}_schaefer_200_7+nsd_train.npz")
train_stimuli = pd.read_csv(train_tsv_path, sep="\t", header=None)[0].values
np.savez_compressed(train_npz_path, beta=train_data, stimuli=train_stimuli, labels=np.array(sorted_labels))

#test
test_schaefer = {}
for label in labels:
    idx = np.where(schaefer_labels == label)[0]
    test_schaefer[int(label)] = beta_test[:, idx]
max_test = max(arr.shape[1] for arr in test_schaefer.values())
print("max_test:", max_test)

padded_test = {}
for label, data in test_schaefer.items():
    pad_width = max_test - data.shape[1]
    if pad_width > 0:
        padded = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
    else:
        padded = data
    padded_test[label] = padded
sorted_labels = sorted(padded_test.keys())
test_data = np.stack([padded_test[label] for label in sorted_labels], axis=0) #shape: (num_labels, num_trials, max_voxels)
test_data = np.transpose(test_data, (1, 0, 2)) #shape: (num_trials, num_labels, max_voxels)
print("Final test_data shape:", test_data.shape)

test_npz_path = os.path.join(beta_dir, f"{SUB}_schaefer_200_7+nsd_test.npz")
test_stimuli = pd.read_csv(test_tsv_path, sep="\t", header=None)[0].values
np.savez_compressed(test_npz_path, beta=test_data, stimuli=test_stimuli, labels=np.array(sorted_labels))