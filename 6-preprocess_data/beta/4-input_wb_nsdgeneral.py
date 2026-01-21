import os
import numpy as np
import nibabel as nib

# ====== config ======
BASE = "/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta"
BETA_ROOT = os.path.join(BASE, "beta_mni_2mm")
OUT_ROOT  = os.path.join(BASE, "connectomind2")

SCH_PATH = os.path.join(BETA_ROOT, "schaefer_mask.nii.gz")

subs = ["sub-01","sub-02","sub-05","sub-07"] 

# ====== load schaefer once ======
sch = nib.load(SCH_PATH).get_fdata(dtype=np.float32)
sch_flat = (sch > 0.5).reshape(-1)
n_sch = int(sch_flat.sum())
print(f"[schaefer] shape={sch.shape} schaefer==1 voxels={n_sch}")

for SUB in subs:
    sub_dir = os.path.join(BETA_ROOT, SUB)

    beta_path = os.path.join(sub_dir, f"{SUB}_beta_train_wb_z.npy")  # (T, n_voxels=132032)
    nsd_path  = os.path.join(sub_dir, f"{SUB}_nsdgeneral_MNI2mm.nii.gz")

    if not os.path.exists(beta_path):
        raise FileNotFoundError(f"beta not found: {beta_path}")
    if not os.path.exists(nsd_path):
        raise FileNotFoundError(f"nsdgeneral not found: {nsd_path}")

    # nsdgeneral load
    nsd = nib.load(nsd_path).get_fdata(dtype=np.float32)
    if nsd.shape != sch.shape:
        raise ValueError(f"[{SUB}] mask shape mismatch: nsdgeneral {nsd.shape} vs schaefer {sch.shape}")

    nsd_flat = (nsd == 1).reshape(-1)

    n_nsd = int(nsd_flat.sum())
    n_common = int((sch_flat & nsd_flat).sum())

    print(f"\n===== {SUB} =====")
    print(f"[mask] nsdgeneral==1 voxels: {n_nsd}")
    print(f"[mask] common==1 voxels (schaefer & nsdgeneral): {n_common}")

    sel = nsd_flat[sch_flat]
    print(f"[mask] selector true: {int(sel.sum())} (should equal common={n_common})")

    # beta: (T, n_sch)
    beta = np.load(beta_path, mmap_mode="r")
    if beta.ndim != 2:
        raise ValueError(f"[{SUB}] beta must be 2D, got {beta.shape}")
    if beta.shape[1] != n_sch:
        raise ValueError(f"[{SUB}] beta second dim != n_sch: beta={beta.shape}, n_sch={n_sch}")

    beta_out = np.asarray(beta[:, sel], dtype=np.float32)  # (T, n_common)

    out_dir = os.path.join(OUT_ROOT, SUB)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{SUB}_beta-train_nsdgeneral.npy")
    np.save(out_path, beta_out)

    print(f"[OK] saved: {out_path}")
    print(f"[OK] out shape: {beta_out.shape}, dtype={beta_out.dtype}")
