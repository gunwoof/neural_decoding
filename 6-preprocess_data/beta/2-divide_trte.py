import os
import glob
import numpy as np
import pandas as pd

SUB = "sub-01"

beta_mni_dir = f"/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni/{SUB}"
responses_tsv = f"/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_nsd/{SUB}/responses.tsv"
stim_csv = "/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_nsd/nsd_stim_info_merged.csv"

out_train = os.path.join(beta_mni_dir, f"{SUB}_beta_train.npy")
out_test  = os.path.join(beta_mni_dir, f"{SUB}_beta_test.npy")

tsv_train = os.path.join(beta_mni_dir, f"{SUB}_beta_train.tsv")
tsv_test  = os.path.join(beta_mni_dir, f"{SUB}_beta_test.tsv")

# ===== 2. beta 파일들 (40개) 순서대로 로드 & concat =====
pattern = os.path.join(beta_mni_dir, f"{SUB}_ses-*_beta_masked.npy")
beta_paths = sorted(glob.glob(pattern))  # ses-01 ~ ses-40 순서로 정렬됨 (zero padding 되어있으니까)

print(f"Found {len(beta_paths)} beta files:")

betas = []
for p in beta_paths:
    arr = np.load(p)   # (101319, 750)
    print(os.path.basename(p), "shape:", arr.shape)
    betas.append(arr)

# (101319, 750) * 40 -> (101319, 30000)
beta_all = np.concatenate(betas, axis=1)
print("Concatenated beta shape:", beta_all.shape)  # 기대: (101319, 30000)

# ===== 3. responses.tsv에서 73KID 읽기 =====
resp = pd.read_csv(responses_tsv, sep="\t")
assert "73KID" in resp.columns, "responses.tsv에 73KID 컬럼이 없음"

kid_seq = resp["73KID"].astype(int).to_numpy()  # 길이 30000
print("len(73KID):", len(kid_seq))

# beta time축 길이와 73KID 길이 일치 확인
assert beta_all.shape[1] == len(kid_seq), "beta time축과 73KID 길이가 다름!"

# ===== 4. nsd_stim_info_merged.csv에서 shared1000 정보 가져오기 =====
stim = pd.read_csv(stim_csv)

# (각 73KID - 1) 을 stim의 num 컬럼에서 찾기
stim_idx = kid_seq - 1

# num을 인덱스로 쓰도록 설정
stim = stim.set_index("num")

# shared1000 컬럼 타입에 따라 bool로 변환
shared_col = stim.loc[stim_idx, "shared1000"]

if shared_col.dtype == bool:
    shared_flags = shared_col.to_numpy()
else:
    # 만약 0/1 이거나 "True"/"False" 문자열이라면 아래 둘 중 하나로 맞춰 쓰면 됨
    # 1) 0/1 이라고 가정:
    try:
        shared_flags = shared_col.astype(int).to_numpy().astype(bool)
    except Exception:
        # 2) "True"/"False" 문자열이라고 가정:
        shared_flags = (shared_col.astype(str) == "True").to_numpy()

print("총 shared1000=True 개수:", shared_flags.sum())
print("총 shared1000=False 개수:", (~shared_flags).sum())

# ===== 5. Train / Test 나누기 =====
# shared1000 == False -> train
# shared1000 == True  -> test
train_mask = ~shared_flags          # 길이 30000
test_mask  = shared_flags

beta_train = beta_all[:, train_mask]   # (101319, 27000) 기대
beta_test  = beta_all[:, test_mask]    # (101319, 3000)  기대

beta_train = beta_train.T
beta_test = beta_test.T

print("beta_train shape:", beta_train.shape)
print("beta_test  shape:", beta_test.shape)

# ===== 6. 저장 =====
np.save(out_train, beta_train)
np.save(out_test, beta_test)

print("Saved:")
print("  train ->", out_train)
print("  test  ->", out_test)

kid_train = kid_seq[train_mask]  # 길이 27000 기대
kid_test  = kid_seq[test_mask]   # 길이 3000  기대

print("kid_train len:", len(kid_train))
print("kid_test  len:", len(kid_test))

# 단일 컬럼 73KID를 갖는 tsv 파일로 저장
pd.DataFrame({"73KID": kid_train}).to_csv(tsv_train, sep="\t", index=False)
pd.DataFrame({"73KID": kid_test}).to_csv(tsv_test,  sep="\t", index=False)

print("Saved tsv:")
print("  train ->", tsv_train)
print("  test  ->", tsv_test)