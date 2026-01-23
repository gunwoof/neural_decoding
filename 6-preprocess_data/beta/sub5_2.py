import re
import numpy as np
from pathlib import Path

sub = "sub-01"
base_dir = Path("/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni_2mm") / sub
tsv_path  = base_dir / f"{sub}_beta_test.tsv"

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
# 1) read tsv (한 번만)
# =========================
lines = [ln.strip() for ln in tsv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
N = len(lines)
if N % 3 != 0:
    raise ValueError(f"TSV line count must be multiple of 3, got {N}")

nums = np.array([parse_num(x) for x in lines], dtype=np.int64)

# =========================
# 2) sort indices (한 번만)
# =========================
order = np.argsort(nums, kind="stable")
nums_sorted  = nums[order]
lines_sorted = [lines[i] for i in order]

# =========================
# 3) validate (한 번만)
# =========================
uniq, counts = np.unique(nums_sorted, return_counts=True)
bad = uniq[counts != 3]
if bad.size > 0:
    raise ValueError(f"These coco nums do NOT appear exactly 3 times: {bad[:20]} ... (showing up to 20)")

# (옵션) sorted tsv 저장: 필요하면 주석 해제
out_tsv_sorted = tsv_path.with_suffix("").as_posix() + "_sorted.tsv"
out_tsv_sorted = Path(out_tsv_sorted)
out_tsv_sorted.write_text("\n".join(lines_sorted) + "\n", encoding="utf-8")
print(f"[OK] wrote sorted tsv: {out_tsv_sorted}")

# =========================
# 4~6) vosdewael{200,300,400}_T 반복
# =========================
chunk = 64  # beta 재정렬 chunk
G = N // 3  # 1000

for n in [200, 300, 400]:
    npy_path = base_dir / f"{sub}_beta_test_vosdewael{n}_T.npy"
    if not npy_path.exists():
        print(f"[SKIP] not found: {npy_path}")
        continue

    out_npy_sorted = npy_path.with_suffix("").as_posix() + "_sorted.npy"
    out_npy_avg    = npy_path.with_suffix("").as_posix() + "_avg.npy"
    out_npy_sorted = Path(out_npy_sorted)
    out_npy_avg    = Path(out_npy_avg)

    # 4) load npy
    beta = np.load(npy_path, mmap_mode="r")  # expected (T, P, V) where T=N
    if beta.ndim != 3:
        raise ValueError(f"Unexpected ndim {beta.ndim} for {npy_path}")
    if beta.shape[0] != N:
        raise ValueError(f"[{n}] npy first dim {beta.shape[0]} != tsv lines {N}")

    T, P, V = beta.shape
    print(f"[INFO] [{n}] beta shape: {beta.shape}")

    # 5) reorder beta according to sorted order (chunk write)
    beta_sorted_mm = np.lib.format.open_memmap(out_npy_sorted, mode="w+", dtype=beta.dtype, shape=beta.shape)
    for start in range(0, T, chunk):
        end = min(T, start + chunk)
        idx = order[start:end]
        beta_sorted_mm[start:end] = beta[idx]
    del beta_sorted_mm
    print(f"[OK] [{n}] wrote sorted npy: {out_npy_sorted}")

    # 6) average each consecutive triplet -> (G, P, V)
    beta_sorted = np.load(out_npy_sorted, mmap_mode="r")
    beta_avg_mm = np.lib.format.open_memmap(out_npy_avg, mode="w+", dtype=np.float32, shape=(G, P, V))

    for g in range(G):
        s = 3 * g
        beta_avg_mm[g] = beta_sorted[s:s+3].astype(np.float32).mean(axis=0)

    del beta_avg_mm
    print(f"[OK] [{n}] wrote averaged npy: {out_npy_avg}")

    # (옵션) unique tsv는 파일 저장 안 하더라도 sanity check만 수행
    unique_lines = [lines_sorted[3*i] for i in range(G)]
    nums_unique = np.array([parse_num(x) for x in unique_lines], dtype=np.int64)
    if not np.all(nums_unique[:-1] <= nums_unique[1:]):
        raise RuntimeError(f"[{n}] Unique order sanity check failed (not ascending).")

print("[DONE] All steps completed successfully.")
