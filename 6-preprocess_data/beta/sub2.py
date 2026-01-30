import os
import numpy as np

sub = "sub-01"
base_dir = f"/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni_2mm/{sub}"

def transpose_save_memmap(in_path, out_path, chunk=256):
    x = np.load(in_path, mmap_mode="r")          # (100, N, V)
    if x.ndim != 3:
        raise ValueError(f"Unexpected ndim {x.ndim} for {in_path}")

    R, N, V = x.shape
    # 출력은 (N, R, V)
    y_mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=x.dtype, shape=(N, R, V))

    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        # x[:, i:j, :] -> (R, chunk, V) 를 (chunk, R, V)로
        y_mm[i:j, :, :] = np.swapaxes(x[:, i:j, :], 0, 1)

    del y_mm  # flush

for n in range(100, 401, 100):
    for split in ["train", "test"]:
        in_path = os.path.join(base_dir, f"{sub}_beta_{split}_vosdewael{n}.npy")
        if not os.path.exists(in_path):
            print(f"[SKIP] not found: {in_path}")
            continue

        out_path = os.path.join(base_dir, f"{sub}_beta_{split}_vosdewael{n}_T.npy")
        transpose_save_memmap(in_path, out_path, chunk=256)

        # shape 확인(헤더만 읽어서 빠름)
        y = np.load(out_path, mmap_mode="r")
        print(f"[OK] {os.path.basename(in_path)} {np.load(in_path, mmap_mode='r').shape} -> {os.path.basename(out_path)} {y.shape}")
