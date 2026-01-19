import numpy as np
import os

SUB = "sub-01"
base_dir = f"/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni/{SUB}"

train_path = os.path.join(base_dir, f"{SUB}_beta_train.npy")
test_path  = os.path.join(base_dir, f"{SUB}_beta_test.npy")

out_train_z = os.path.join(base_dir, f"{SUB}_beta_train_z.npy")
out_test_z  = os.path.join(base_dir, f"{SUB}_beta_test_z.npy")

# 1) train / test 로드
beta_train = np.load(train_path)   # (27000, 101319)
beta_test  = np.load(test_path)    # (3000, 101319)

print("beta_train:", beta_train.shape)
print("beta_test :", beta_test.shape)

# 2) train 기준 voxel-wise 평균 / 표준편차 계산 (각 voxel = column 기준)
mean_vox = beta_train.mean(axis=0)          # (101319,)
std_vox  = beta_train.std(axis=0, ddof=0)   # (101319,)

# 분산 0인 voxel 대비 (std=0이면 1로 치환)
eps = 1e-8
std = np.where(std_vox < eps, 1.0, std_vox)

# 3) z-score 정규화: (X - mean) / std  (train mean, std로 train, test 둘 다 정규화)
beta_train_z = (beta_train - mean_vox) / std   # (27000, 101319)
beta_test_z  = (beta_test  - mean_vox) / std   # (3000, 101319)

print("beta_train_z:", beta_train_z.shape)
print("beta_test_z :", beta_test_z.shape)

# 4) 저장
np.save(out_train_z, beta_train_z)
np.save(out_test_z, beta_test_z)

print("Saved:")
print("  ", out_train_z)
print("  ", out_test_z)
