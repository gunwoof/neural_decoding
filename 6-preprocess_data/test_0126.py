#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib

SUB = "sub-01"

train_npy = f"/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_huggingface/{SUB}/{SUB}_nsdgeneral_train.npy"
test_npy  = f"/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_huggingface/{SUB}/{SUB}_nsdgeneral_test.npy"

nsd_mask_nii = f"/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_huggingface/{SUB}/{SUB}_mask-nsdgeneral.nii"
dk_mask_nii  = f"/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_huggingface/{SUB}/{SUB}_mask-dk.nii"
out_dir = f"/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/connectomind1_v2/{SUB}"
out_train = os.path.join(out_dir, f"{SUB}_beta-train_dk.npy")
out_test  = os.path.join(out_dir, f"{SUB}_beta-test_dk.npy")

def load_mask_flat(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data.reshape(-1)

def build_dk_mapping(nsd_mask_flat, dk_mask_flat):
    # nsdgeneral==1인 위치 (원본 볼륨 flatten index)
    nsd_idx = np.where(nsd_mask_flat == 1)[0]
    if nsd_idx.size == 0:
        raise RuntimeError("nsdgeneral mask has no 1-voxels.")

    # nsd subset 안에서의 DK 라벨
    dk_labels = dk_mask_flat[nsd_idx]
    dk_labels = np.asarray(np.rint(dk_labels), dtype=np.int32)

    keep = ((1000 <= dk_labels) & (dk_labels <= 1035)) | ((2000 <= dk_labels) & (dk_labels <= 2035))

    # keep 아닌 곳은 0으로 만들어서 parcel에서 제외되게
    dk_labels_filtered = dk_labels.copy()
    dk_labels_filtered[~keep] = 0

    parcel_labels = np.unique(dk_labels_filtered)
    parcel_labels = parcel_labels[parcel_labels > 0]
    if parcel_labels.size == 0:
        raise RuntimeError("No DK parcel labels in [1000~1035] or [2000~2035] within nsdgeneral==1 voxels.")

    parcel_to_cols = []
    voxel_counts = []
    for lab in parcel_labels:
        cols = np.where(dk_labels_filtered == lab)[0]  # subset column indices
        parcel_to_cols.append(cols.astype(np.int32, copy=False))
        voxel_counts.append(cols.size)

    maxV = int(np.max(voxel_counts))
    return parcel_labels.astype(np.int32), parcel_to_cols, maxV, np.array(voxel_counts, dtype=np.int32)


def parcellate_pad_memmap(in_npy, out_npy, parcel_to_cols, maxV, chunk_trials=64):
    X = np.load(in_npy, mmap_mode="r")  # (Ntrial, Vsubset)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array (Ntrial,V). Got shape={X.shape} from {in_npy}")
    N, V = X.shape

    P = len(parcel_to_cols)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)

    # 출력: (Ntrial, Nparcel, maxV) 를 memmap으로 생성
    Y = np.lib.format.open_memmap(out_npy, mode="w+", dtype=X.dtype, shape=(N, P, maxV))
    Y[:] = 0  # zero padding 기본값

    # parcel 단위로 쓰면 랜덤 IO가 커질 수 있어 trial chunk로 씀
    for p, cols in enumerate(parcel_to_cols):
        nv = cols.size
        if nv == 0:
            continue
        for i in range(0, N, chunk_trials):
            j = min(i + chunk_trials, N)
            # (chunk, nv) -> (chunk, maxV) 앞쪽에만 채움
            Y[i:j, p, :nv] = X[i:j, cols]

    # flush
    del Y

def main():
    # 1) 마스크 로드 & 매핑 생성
    nsd_flat = load_mask_flat(nsd_mask_nii)
    dk_flat  = load_mask_flat(dk_mask_nii)

    if nsd_flat.shape != dk_flat.shape:
        raise ValueError(f"Mask shape mismatch: nsd={nsd_flat.shape}, dk={dk_flat.shape}")

    parcel_labels, parcel_to_cols, maxV, voxel_counts = build_dk_mapping(nsd_flat, dk_flat)

    print(f"[INFO] #parcels={len(parcel_labels)}  max_voxels_per_parcel={maxV}")
    print(f"[INFO] voxel_counts(min/median/mean/max) = "
          f"{voxel_counts.min()} / {np.median(voxel_counts)} / {voxel_counts.mean():.2f} / {voxel_counts.max()}")


    # 2) train/test parcellation + padding
    parcellate_pad_memmap(train_npy, out_train, parcel_to_cols, maxV, chunk_trials=64)
    print(f"[OK] saved: {out_train}")

    parcellate_pad_memmap(test_npy, out_test, parcel_to_cols, maxV, chunk_trials=128)
    print(f"[OK] saved: {out_test}")

    # 3) shape 확인
    Ytr = np.load(out_train, mmap_mode="r")
    Yte = np.load(out_test,  mmap_mode="r")
    print(f"[CHECK] train out shape: {Ytr.shape} dtype={Ytr.dtype}")
    print(f"[CHECK] test  out shape: {Yte.shape} dtype={Yte.dtype}")

if __name__ == "__main__":
    main()
