import os, re, shutil, tarfile
import numpy as np
import pandas as pd
import nibabel as nib
from huggingface_hub import hf_hub_download
from glob import glob
from tqdm import tqdm

### download beta from huggingface and unzip ###

root_dir = "/nas/research/03-Neural_decoding/1-raw_data" # change to your path to downlad raw data
train_dir = os.path.join(root_dir, "beta-train")
test_dir = os.path.join(root_dir, "beta-test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

repo_id = "pscotti/naturalscenesdataset"

files = [
    *[f"webdataset_avg_split/train/train_subj01_{i:02d}.tar" for i in range(18)],
    "webdataset_avg_split/val/val_subj01_0.tar",
    "webdataset_avg_split/test/test_subj01_0.tar",
    "webdataset_avg_split/test/test_subj01_1.tar",
]

for f in files:
    
    if "/test/" in f:
        save_dr = test_dir
    else:
        save_dr = train_dir

    tar_path = hf_hub_download(
        repo_id=repo_id, 
        filename=f, 
        local_dir=save_dr,
        local_dir_use_symlinks=False
    )
    print("Downloaded:", tar_path)
    
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=save_dr)
    print("Extracted to:", save_dr)


### beta file들 합치기 ###

beta_dir = "/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta"
os.makedirs(beta_dir, exist_ok=True)

beta_train_path = os.path.join(beta_dir,"sub-01_nsdgeneral_train.npy")
beta_test_path = os.path.join(beta_dir,"sub-01_nsdgeneral_test.npy")

train_list = sorted(glob(os.path.join(train_dir, '*nsdgeneral.npy')))
print(f"총 {len(train_list)}개의 파일")

train_arr = []

for f in train_list:
    data = np.load(f)
    train_arr.append(data)

if train_arr:
    concat_train = np.concatenate(train_arr, axis=0)
    np.save(beta_train_path, concat_train)
    print(f"Saved train npy file to: {beta_train_path}")
else:
    print("No training data found.")

test_list  = sorted(glob(os.path.join(test_dir,  '*nsdgeneral.npy')))
print(f"총 {len(test_list)}개의 파일")

test_arr = []

for f in test_list:
    data = np.load(f)
    test_arr.append(data)

if test_arr:
    concat_test = np.concatenate(test_arr, axis=0)
    np.save(beta_test_path, concat_test)
    print(f"Saved test npy file to: {beta_test_path}")
else:
    print("No test data found.")


###대응 tsv 파일 만들기 ###

tsv_dir = "/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta"
os.makedirs(tsv_dir, exist_ok=True)

train_tsv_path = os.path.join(tsv_dir, "sub-01_mapped_img_train.tsv")
test_tsv_path = os.path.join(tsv_dir, "sub-01_mapped_img_test.tsv")

coco_files_train = sorted(glob(os.path.join(train_dir, 'sample?????????.coco73k.npy')))
train_filenames = []

print("Creating mapped_img_train.tsv...")

for coco_path in tqdm(coco_files_train):
    try:
        value = int(np.load(coco_path))
        new_name = f"coco2017_{value+1}.jpg"
        train_filenames.extend([new_name] * 3)

    except Exception as e:
        print(f"Error processing {coco_path}: {e}")

df_train = pd.DataFrame(train_filenames)
df_train.to_csv(train_tsv_path, sep="\t", header=False, index=False)

print(f"Saved train tsv to: {train_tsv_path}")
print(f"mapped_img_train shape: {df_train.shape}")

coco_files_test = sorted(glob(os.path.join(test_dir, 'sample?????????.coco73k.npy')))
test_filenames = []

print("Creating mapped_img_test.tsv...")

for coco_path in tqdm(coco_files_test):
    try:
        value = int(np.load(coco_path))
        new_name = f"coco2017_{value+1}.jpg"
        test_filenames.append(new_name)

    except Exception as e:
        print(f"Error processing {coco_path}: {e}")

df_test = pd.DataFrame(test_filenames)
df_test.to_csv(test_tsv_path, sep="\t", header=False, index=False)

print(f"Saved test tsv to: {test_tsv_path}")
print(f"mapped_img_test shape: {df_test.shape}")


### image 폴더에 모으기 ###

img_dr = "/nas/research/03-Neural_decoding/4-image/beta"
os.makedirs(img_dr, exist_ok=True)

for npy_f in coco_files_train:
    try:
        basename = os.path.basename(npy_f).replace('.coco73k.npy', '')
        jpg_path = os.path.join(img_dr, f"{basename}.jpg")

        value = int(np.load(npy_f))
        new_name = f"coco2017_{value+1}.jpg"
        out_path = os.path.join(img_dr, new_name)

        if os.path.exists(jpg_path):
            shutil.copy(jpg_path, out_path)
        else:
            print(f"⚠️ JPG file not found for {npy_f}")
    except Exception as e:
        print(f"❌ Error processing {npy_f}: {e}")


for npy_f in coco_files_test:
    try:
        basename = os.path.basename(npy_f).replace('.coco73k.npy', '')
        jpg_path = os.path.join(img_dr, f"{basename}.jpg")

        value = int(np.load(npy_f))
        new_name = f"coco2017_{value+1}.jpg"
        out_path = os.path.join(img_dr, new_name)

        if os.path.exists(jpg_path):
            shutil.copy(jpg_path, out_path)
        else:
            print(f"⚠️ JPG file not found for {npy_f}")
    except Exception as e:
        print(f"❌ Error processing {npy_f}: {e}")


### beta 파일에 nsdgeneral과 desikan-killiany 마스크 이용해서 마스킹 -> 20개의 label로 나뉨 ###

nsd_mask_path = "/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_hf_dk/connectomind/sub01_nsdgeneral.nii"
dk_mask_path = "/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_hf_dk/connectomind/sub-01_dk.nii"

nsd_mask = nib.load(nsd_mask_path).get_fdata().flatten()
dk_mask = nib.load(dk_mask_path).get_fdata().flatten()

beta_train = np.load(beta_train_path)
beta_test = np.load(beta_test_path)

valid_idx = np.where(nsd_mask ==1)[0]
dk_labels = dk_mask[valid_idx]

cortical_labels = [v for v in np.unique(dk_labels) if 1000 <= v <= 1035 or 2000 <= v <= 2035]

train_dk = {}

for label in cortical_labels:
    idx = np.where(dk_labels == label)[0]
    train_dk[int(label)] =  beta_train[:, idx]

max_train = max(arr.shape[1] for arr in train_dk.values())

padded_train_dk = {}
for label, data in train_dk.items():
    original_shape = data.shape
    pad_width = max_train - data.shape[1]
    if pad_width > 0:
        padded = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
    else:
        padded = data
    padded_train_dk[label] = padded

train_data = np.stack([padded_train_dk[label] for label in sorted(padded_train_dk.keys())], axis=0)
train_data = np.transpose(train_data, (1, 0, 2))
np.save(os.path.join(beta_dir, "sub-01_dk_train.npy"), train_data)

test_dk = {}

for label in cortical_labels:
    idx = np.where(dk_labels == label)[0]
    test_dk[int(label)] =  beta_test[:, idx]

max_test = max(arr.shape[1] for arr in test_dk.values())

padded_test_dk = {}
for label, data in test_dk.items():
    original_shape = data.shape
    pad_width = max_test - data.shape[1]
    if pad_width > 0:
        padded = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
    else:
        padded = data
    padded_test_dk[label] = padded

test_data = np.stack([padded_test_dk[label] for label in sorted(padded_test_dk.keys())], axis=0)
test_data = np.transpose(test_data, (1, 0, 2))
np.save(os.path.join(beta_dir, "sub-01_dk_test.npy"), test_data)


### beta.npy 파일과 img 파일 합쳐 npz 파일로 ###

train_npz_path = os.path.join(beta_dir, "sub-01_train.npz")
test_npz_path = os.path.join(beta_dir, "sub-01_test.npz")

train_img = pd.read_csv(train_tsv_path, sep="\t", header=None)[0].values
test_img = pd.read_csv(test_tsv_path, sep="\t", header=None)[0].values

assert train_data.shape[0] == len(train_img), "Train data and image count mismatch"
assert test_data.shape[0] == len(test_img), "Test data and image count mismatch"

np.savez_compressed(train_npz_path, X = train_data, Y = train_img)
np.savez_compressed(test_npz_path, X = test_data, Y = test_img)

print(f"Saved train npz to: {train_npz_path}")
print(f"Saved test npz to: {test_npz_path}")
