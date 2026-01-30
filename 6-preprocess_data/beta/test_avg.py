import re
import numpy as np
from pathlib import Path

# =========================
# inputs
# =========================
SUB = "sub-02"
base_dir = Path("/research/wsi/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/connectomind2") / SUB
tsv_path = Path("/research/wsi/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni_2mm") / SUB / f"{SUB}_beta_test.tsv"

vosdewael_list = list(range(100, 401, 100))  

# =========================
# helper: parse coco num
# =========================
pat = re.compile(r"coco2017_(\d+)\.jpg$")

def parse_num(s: str) -> int:
    s = s.strip()
    m = pat.search(s)
    if not m:
        raise ValueError(f"Unexpected tsv entry (not coco2017_{{num}}.jpg): {s}")
    return int(m.group(1))

# =========================
# 1) read tsv -> build stable sort order (no writing)
# =========================
lines = [ln.strip() for ln in tsv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
N = len(lines)
if N % 3 != 0:
    raise ValueError(f"TSV line count must be multiple of 3, got {N}")

nums = np.array([parse_num(x) for x in lines], dtype=np.int64)

order = np.argsort(nums, kind="stable")  # stable: keep relative order within same num

nums_sorted = nums[order]
uniq, counts = np.unique(nums_sorted, return_counts=True)
bad = uniq[counts != 3]
if bad.size > 0:
    raise ValueError(f"These coco nums do NOT appear exactly 3 times: {bad[:20]} ...")

T = N
G = T // 3
print(f"[INFO] TSV lines={T}, unique groups={G}")

# =========================
# per-vosdewael: reorder-on-the-fly -> avg only
# =========================
def process_vosdewael(n: int, chunk_groups: int = 8):
    npy_path = base_dir / f"{SUB}_beta-test_vosdewael{n}_3000.npy"
    if not npy_path.exists():
        print(f"[SKIP] missing: {npy_path}")
        return

    beta = np.load(npy_path, mmap_mode="r")  # (T, P, V)
    if beta.shape[0] != T:
        raise ValueError(f"[vosdewael{n}] npy first dim {beta.shape[0]} != tsv lines {T}")

    _, P, V = beta.shape
    out_npy_avg = base_dir / f"{SUB}_beta-test_vosdewael{n}.npy"

    print(f"[INFO] vosdewael{n}: beta shape={beta.shape} -> avg shape=({G},{P},{V})")

    beta_avg_mm = np.lib.format.open_memmap(
        out_npy_avg, mode="w+", dtype=np.float32, shape=(G, P, V)
    )

    # group 단위로 끊어서 reorder + mean
    for g0 in range(0, G, chunk_groups):
        g1 = min(G, g0 + chunk_groups)

        # 이 chunk에 해당하는 원본 trial indices (길이 3*(g1-g0))
        ord_chunk = order[3*g0 : 3*g1]  # length = 3*(g1-g0)

        # (3*(ng), P, V) 로 가져온 뒤 (ng, 3, P, V)로 reshape
        block = beta[ord_chunk].astype(np.float32)  # load + cast
        ng = (g1 - g0)
        block = block.reshape(ng, 3, P, V).mean(axis=1)  # (ng, P, V)

        beta_avg_mm[g0:g1] = block

    del beta_avg_mm  # flush
    print(f"[OK] vosdewael{n}: wrote avg npy: {out_npy_avg}")

for n in vosdewael_list:
    process_vosdewael(n)

print("[DONE] avg-only for vosdewael100~400 (missing ones skipped).")
