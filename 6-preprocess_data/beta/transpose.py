import os
import numpy as np

sub = "sub-01"
base_dir = f"/research/wsi/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni_2mm/{sub}"


for n in range(200, 1001, 100):
    for split in ["train", "test"]:
        in_path = os.path.join(base_dir, f"{sub}_beta_{split}_schaefer{n}.npy")
        if not os.path.exists(in_path):
            print(f"[SKIP] not found: {in_path}")
            continue

        x = np.load(in_path)
        if x.ndim != 3:
            raise ValueError(f"Unexpected ndim {x.ndim} for {in_path}")

        y = np.transpose(x, (1, 0, 2))

        out_path = os.path.join(base_dir, f"{sub}_beta_{split}_schaefer{n}_T.npy")
        np.save(out_path, y)

        print(f"[OK] {os.path.basename(in_path)} {x.shape} -> {os.path.basename(out_path)} {y.shape}")
