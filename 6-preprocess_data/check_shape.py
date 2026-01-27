#!/usr/bin/env python3
import os
import numpy as np

ROOT = "/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_huggingface/sub-02"

for dirpath, dirnames, filenames in os.walk(ROOT):
    npy_files = sorted([f for f in filenames if f.endswith(".npy")])
    if not npy_files:
        continue

    print(f"\n=== {dirpath} ===")

    for fn in npy_files:
        fp = os.path.join(dirpath, fn)
        try:
            arr = np.load(fp, allow_pickle=True)
            print(f"{fn}\tshape={arr.shape}\tdtype={arr.dtype}")
        except Exception as e:
            print(f"{fn}\t[ERROR] {e}")
