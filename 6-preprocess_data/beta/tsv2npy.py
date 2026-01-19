import numpy as np
from pathlib import Path

tsv_path = Path("/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni_2mm/sub-07/sub-07_beta_test_sorted_unique.tsv")
out_npy  = tsv_path.with_suffix(".npy")  # .../sub-07_beta_test_sorted_unique.npy

# TSV 읽기 (빈 줄 제거)
lines = [ln.strip() for ln in tsv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

# 문자열 배열로 저장 (pickle 없이 깔끔하게)
arr = np.array(lines, dtype=object)
np.save(out_npy, arr)

print(f"[OK] saved: {out_npy} shape={arr.shape} dtype={arr.dtype}")
