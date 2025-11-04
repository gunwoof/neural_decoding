import os, re, glob
import numpy as np
import pandas as pd
import nibabel as nib

# ----- 경로 설정 -----
BASE = "/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_nsd/sub-01"
nii_dir = BASE
responses_tsv = os.path.join(BASE, "responses.tsv")
stim_csv = os.path.join(BASE, "subject1_stim_info.csv")

out_train = os.path.join(BASE, "sub-01_beta_train.nii.gz")
out_test  = os.path.join(BASE, "sub-01_beta_test.nii.gz")

# 마스크 경로 (nsdgeneral: 값 1/0/-1 → 여기서는 '정확히 1'만 남김)
mask_path = os.path.join(BASE, "nsdgeneral.nii.gz")

# ----- 세션 파일 수집 & 자연 정렬 -----
paths = glob.glob(os.path.join(nii_dir, "betas_session*.nii.gz"))

def session_key(p):
    m = re.search(r"betas_session(\d+)\.nii\.gz$", os.path.basename(p))
    return int(m.group(1)) if m else 10**9

paths = sorted(paths, key=session_key)
assert len(paths) > 0, "betas_session*.nii.gz 파일을 찾지 못했습니다."

# ----- responses.tsv: 73KID 읽기 (길이 30000 가정) -----
resp = pd.read_csv(responses_tsv, sep="\t")
assert "73KID" in resp.columns, "responses.tsv에 73KID 컬럼이 없습니다."
kid_seq = resp["73KID"].astype(int).to_numpy()
N_total = len(kid_seq)
print(f"[INFO] responses 73KID 개수: {N_total}")

# ----- subject1_stim_info.csv: num -> shared1000(bool) 매핑 -----
stim = pd.read_csv(stim_csv)
assert "num" in stim.columns and "shared1000" in stim.columns, "subject1_stim_info.csv에 num/shared1000 컬럼이 없습니다."
shared_map = dict(
    zip(
        stim["num"].astype(int),
        stim["shared1000"].astype(str).str.lower().isin(["1","true","t","yes"])
    )
)

# ----- 73KID 순서대로 train/test 레이블 배열 만들기 -----
cat = np.empty(N_total, dtype=np.int8)  # 0(train) / 1(test)
for i, k in enumerate(kid_seq):
    v = shared_map.get(k, None)
    if v is None:
        raise ValueError(f"subject1_stim_info.csv에 num={k}가 없습니다.")
    cat[i] = 1 if v else 0

n_train = int((cat == 0).sum())
n_test  = int((cat == 1).sum())
print(f"[INFO] 분할 예측: train={n_train}, test={n_test}")
if n_train != 27000 or n_test != 3000:
    print(f"[WARN] 기대치(27000/3000)와 다름: train={n_train}, test={n_test}")

# ----- 첫 파일에서 공간 정보 가져오기 -----
first_img = nib.load(paths[0])
affine = first_img.affine
hdr = first_img.header.copy()
first_shape = first_img.shape  # (X,Y,Z,T_sess)
X, Y, Z = first_shape[:3]
print(f"[INFO] 공간 크기: {(X,Y,Z)}")

# ----- 마스크 로드 및 검증 (값==1만 True) -----
mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata(dtype=np.float32)  # 메모리에 올려도 보통 마스크는 작음
assert mask_data.shape[:3] == (X, Y, Z), f"마스크 공간 크기 불일치: {mask_data.shape[:3]} vs {(X,Y,Z)}"
mask = (mask_data == 1)  # 정확히 1인 위치만 사용
mask = mask.astype(np.bool_)  # bool로 고정
del mask_data
print(f"[INFO] 마스크 로드 완료: True(1) voxel 수 = {int(mask.sum())}")

# ----- 출력 NIfTI를 메모리 절약형으로 만들기 -----
train_mm = np.memmap(os.path.join(BASE, "._tmp_train_mm.dat"), mode="w+", dtype=np.float32, shape=(X, Y, Z, n_train))
test_mm  = np.memmap(os.path.join(BASE, "._tmp_test_mm.dat"),  mode="w+", dtype=np.float32, shape=(X, Y, Z, n_test))

train_ptr = 0
test_ptr = 0
global_ptr = 0

# ----- 세션 파일 순회하며 스트리밍 복사 (마스킹 선적용) -----
for p in paths:
    img = nib.load(p)
    dataobj = img.dataobj  # 지연 로딩
    assert dataobj.shape[:3] == (X, Y, Z), f"공간 크기 불일치: {p}"
    T_local = dataobj.shape[3] if dataobj.ndim == 4 else 1

    # 세션 로그 집계용
    sess_name = os.path.basename(p).replace(".nii.gz", "")
    sess_train_add = 0
    sess_test_add  = 0

    for t in range(T_local):
        if global_ptr >= N_total:
            raise RuntimeError(f"73KID 총 개수({N_total})보다 세션 볼륨이 더 많습니다. 마지막 파일: {p}")

        label = cat[global_ptr]  # 0 or 1

        # (X,Y,Z) 볼륨 단위로 로드 → 마스크 적용: mask==1 위치만 값 유지, 나머지는 0
        vol = np.asanyarray(dataobj[..., t], dtype=np.float32)
        # in-place로 마스킹 (불필요한 복사 방지)
        vol[~mask] = 0.0

        if label == 0:
            train_mm[..., train_ptr] = vol
            train_ptr += 1
            sess_train_add += 1
        else:
            test_mm[..., test_ptr] = vol
            test_ptr += 1
            sess_test_add += 1

        global_ptr += 1

    # 세션 단위 진행 로그
    print(f"[SESSION DONE] {sess_name} → train +{sess_train_add}, test +{sess_test_add} "
          f"(누적: train={train_ptr}/{n_train}, test={test_ptr}/{n_test}, global={global_ptr}/{N_total})")

# ----- 개수 검증 -----
assert global_ptr == N_total, f"총 볼륨 수 불일치: 채운={global_ptr}, 기대={N_total}"
assert train_ptr == n_train and test_ptr == n_test, f"train/test 채움 개수 불일치: train={train_ptr}/{n_train}, test={test_ptr}/{n_test}"

# ----- NIfTI 저장 -----
hdr.set_data_dtype(np.float32)
img_train = nib.Nifti1Image(train_mm, affine, header=hdr)
img_test  = nib.Nifti1Image(test_mm,  affine, header=hdr)
nib.save(img_train, out_train)
nib.save(img_test,  out_test)

print("[DONE]")
print(f"Saved: {out_train}  shape={(X,Y,Z,n_train)}")
print(f"Saved: {out_test}   shape={(X,Y,Z,n_test)}")
