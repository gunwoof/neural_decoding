import re
import numpy as np
from pathlib import Path

# -------------------------
# paths
# -------------------------
beta_path = Path("/research/wsi/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/connectomind2/sub-07/sub-07_beta-test_wb.npy")
tsv_path  = Path("/research/wsi/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni_2mm/sub-07/sub-07_beta_test.tsv")
out_path  = beta_path.parent / "sub-07_beta-test_wb_avg.npy"

# -------------------------
# load
# -------------------------
beta = np.load(beta_path)  # (3000, 17355)
lines = [ln.strip() for ln in tsv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

if beta.shape[0] != len(lines):
    raise ValueError(f"Row mismatch: beta rows={beta.shape[0]} vs tsv lines={len(lines)}")

# -------------------------
# parse coco2017_{num}.jpg -> num
# -------------------------
pat = re.compile(r"coco2017_(\d+)\.jpg$")
nums = np.empty(len(lines), dtype=np.int64)

for i, s in enumerate(lines):
    m = pat.search(s)
    if not m:
        raise ValueError(f"Unexpected TSV entry at line {i+1}: {s}")
    nums[i] = int(m.group(1))

# -------------------------
# sort by num (ascending) and reorder beta accordingly
# -------------------------
order = np.argsort(nums, kind="stable")
nums_sorted = nums[order]
beta_sorted = beta[order]

# -------------------------
# sanity checks (optional but recommended)
# 1) should be divisible by 3
# 2) after sorting, every 3 should be same num
# -------------------------
N = beta_sorted.shape[0]
if N % 3 != 0:
    raise ValueError(f"Total rows {N} is not divisible by 3")

trip = nums_sorted.reshape(-1, 3)
if not np.all(trip[:, 0] == trip[:, 1]) or not np.all(trip[:, 1] == trip[:, 2]):
    # 이 경우는 "각 이미지가 정확히 3번씩" 구조가 깨진 것
    bad = np.where(~((trip[:, 0] == trip[:, 1]) & (trip[:, 1] == trip[:, 2])))[0][:10]
    raise ValueError(f"Not grouped into exact triples after sorting. Example bad groups idx: {bad}")

# -------------------------
# average every 3 rows -> (1000, 17355)
# -------------------------
beta_avg = beta_sorted.reshape(-1, 3, beta_sorted.shape[1]).mean(axis=1)

# -------------------------
# save
# -------------------------
np.save(out_path, beta_avg)
print(f"[OK] saved: {out_path}")
print(f"     avg shape={beta_avg.shape}, dtype={beta_avg.dtype}")
print(f"     unique images={beta_avg.shape[0]} (expected {N//3})")
