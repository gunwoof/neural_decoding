import re
import numpy as np
from pathlib import Path

# =========================
# inputs
# =========================
npy_path = Path("/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni_2mm/sub-07/sub-07_beta_test_schaefer100_T.npy")
tsv_path = Path("/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni_2mm/sub-07/sub-07_beta_test.tsv")

# =========================
# outputs
# =========================
out_tsv_sorted = tsv_path.with_suffix("").as_posix() + "_sorted.tsv"
out_npy_sorted = npy_path.with_suffix("").as_posix() + "_sorted.npy"
out_npy_avg    = npy_path.with_suffix("").as_posix() + "_avg.npy"

out_tsv_sorted = Path(out_tsv_sorted)
out_npy_sorted = Path(out_npy_sorted)
out_npy_avg    = Path(out_npy_avg)

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
# 1) read tsv
# =========================
lines = [ln.strip() for ln in tsv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
N = len(lines)
if N % 3 != 0:
    raise ValueError(f"TSV line count must be multiple of 3, got {N}")

nums = np.array([parse_num(x) for x in lines], dtype=np.int64)

# =========================
# 2) sort indices by (num asc, then keep stable within same num)
# =========================
# stable sort so that within 동일 num의 3개는 기존 상대순서를 유지
order = np.argsort(nums, kind="stable")
nums_sorted = nums[order]
lines_sorted = [lines[i] for i in order]

# =========================
# 3) validate: each num appears exactly 3 times, and becomes consecutive blocks
# =========================
uniq, counts = np.unique(nums_sorted, return_counts=True)
bad = uniq[counts != 3]
if bad.size > 0:
    raise ValueError(f"These coco nums do NOT appear exactly 3 times: {bad[:20]} ... (showing up to 20)")


# write sorted tsv
out_tsv_sorted.write_text("\n".join(lines_sorted) + "\n", encoding="utf-8")
print(f"[OK] wrote sorted tsv: {out_tsv_sorted}")

# =========================
# 4) load npy (memmap to avoid huge RAM)
# =========================
beta = np.load(npy_path, mmap_mode="r")  # shape (3000, 200, #vox)
if beta.shape[0] != N:
    raise ValueError(f"npy first dim {beta.shape[0]} != tsv lines {N}")

T, P, V = beta.shape  # T=3000, P=200
print(f"[INFO] beta shape: {beta.shape}")

# =========================
# 5) reorder beta according to sorted order
# =========================
# NOTE: fancy indexing on memmap can still allocate; to be safe, write in chunks.
beta_sorted_mm = np.lib.format.open_memmap(out_npy_sorted, mode="w+", dtype=beta.dtype, shape=beta.shape)

chunk = 64  # 조절 가능 (메모리 상황에 따라)
for start in range(0, T, chunk):
    end = min(T, start + chunk)
    idx = order[start:end]
    beta_sorted_mm[start:end] = beta[idx]
del beta_sorted_mm  # flush
print(f"[OK] wrote sorted npy: {out_npy_sorted}")

# =========================
# 6) average each consecutive triplet -> (1000, 200, #vox)
# =========================
beta_sorted = np.load(out_npy_sorted, mmap_mode="r")  # now already sorted
G = T // 3  # 1000
beta_avg_mm = np.lib.format.open_memmap(out_npy_avg, mode="w+", dtype=np.float32, shape=(G, P, V))

# 평균은 float32로 저장(원하면 beta.dtype로 바꿔도 됨)
for g in range(G):
    s = 3 * g
    # (3, 200, V) -> mean over axis=0 => (200, V)
    beta_avg_mm[g] = beta_sorted[s:s+3].astype(np.float32).mean(axis=0)

del beta_avg_mm
print(f"[OK] wrote averaged npy (avg over each 3): {out_npy_avg}")

# =========================
# 7) also write unique tsv (1000 lines) if you want
# =========================
# 각 3개 블록의 대표(동일 파일명)만 남긴 tsv
out_tsv_unique = out_tsv_sorted.with_name(out_tsv_sorted.stem + "_unique.tsv")
unique_lines = [lines_sorted[3*i] for i in range(G)]
out_tsv_unique.write_text("\n".join(unique_lines) + "\n", encoding="utf-8")
print(f"[OK] wrote unique tsv (1000 lines): {out_tsv_unique}")

# final sanity
nums_unique = np.array([parse_num(x) for x in unique_lines], dtype=np.int64)
if not np.all(nums_unique[:-1] <= nums_unique[1:]):
    raise RuntimeError("Unique TSV is not sorted ascending — unexpected.")
print("[DONE] All steps completed successfully.")
