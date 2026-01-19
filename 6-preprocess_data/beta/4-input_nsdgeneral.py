import os
import numpy as np
import pandas as pd

SUB = "sub-01"
base_dir = f"/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni/{SUB}"

# 파일 경로
train_z_path = os.path.join(base_dir, f"{SUB}_beta_train_z.npy")
test_z_path  = os.path.join(base_dir, f"{SUB}_beta_test_z.npy")
train_tsv_path = os.path.join(base_dir, f"{SUB}_beta_train.tsv")
test_tsv_path  = os.path.join(base_dir, f"{SUB}_beta_test.tsv")

train_npz_path = os.path.join(base_dir, f"{SUB}_train_nsdgeneral.npz")
test_npz_path  = os.path.join(base_dir, f"{SUB}_test_nsdgeneral.npz")


def load_and_make_filenames(tsv_path):
    vals = pd.read_csv(tsv_path, sep="\t", header=None).iloc[:, 0].astype(str).to_numpy()

    # 헤더 제거 (이미 pandas로 한번 저장해서 'filename'이 생긴 경우 방지)
    if vals[0].lower() in ("73kid", "filename"):
        vals = vals[1:]

    # 전부 숫자이면 73KID로 간주하고 파일명으로 변환
    if np.all([v.isdigit() for v in vals]):
        filenames = np.array([f"coco2017_{int(v)}.jpg" for v in vals], dtype=object)
    else:
        # 이미 파일명이라고 보고 그대로 사용
        filenames = vals

    return filenames


# 1) TSV 읽고 → coco 파일명 배열로 만들기
train_filenames = load_and_make_filenames(train_tsv_path)
test_filenames  = load_and_make_filenames(test_tsv_path)

print("Train stimuli (filenames) shape:", train_filenames.shape)
print("Test stimuli  (filenames) shape:", test_filenames.shape)

# 2) TSV를 헤더/인덱스 없이 다시 저장 (컬럼 이름 절대 안 생김)
pd.DataFrame(train_filenames).to_csv(train_tsv_path, sep="\t", index=False, header=False)
pd.DataFrame(test_filenames).to_csv(test_tsv_path,  sep="\t", index=False, header=False)

print("✔ TSV 변환 및 저장 완료 (헤더 없음)")
print("  train ->", train_tsv_path)
print("  test  ->", test_tsv_path)

# 3) NPY 불러오기
train_z = np.load(train_z_path)
test_z  = np.load(test_z_path)

print("train_z:", train_z.shape)
print("test_z :", test_z.shape)

# sanity check (옵션) : trial 수와 stimuli 길이 일치 확인
assert train_z.shape[0] == train_filenames.shape[0], "train_z 첫 축과 train TSV 길이가 다름"
assert test_z.shape[0]  == test_filenames.shape[0],  "test_z 첫 축과 test TSV 길이가 다름"

# 4) NPZ로 저장 (stimuli = 이미지 파일 이름)
np.savez(train_npz_path, beta=train_z, stimuli=train_filenames)
np.savez(test_npz_path,  beta=test_z,  stimuli=test_filenames)

print("✔ NPZ 저장 완료")
print("  train ->", train_npz_path)
print("  test  ->", test_npz_path)
