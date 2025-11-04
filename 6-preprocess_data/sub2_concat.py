import os, re, glob
import numpy as np
import pandas as pd
import nibabel as nib

# ----- 경로 설정 -----
BASE = "/nas/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_nsd/sub-02"
nii_dir = BASE
responses_tsv = os.path.join(BASE, "responses.tsv")
stim_csv = os.path.join(BASE, "subject2_stim_info.csv")

out_train = os.path.join(BASE, "sub-02_beta_train.nii.gz")
out_test  = os.path.join(BASE, "sub-02_beta_test.nii.gz")

# ----- 세션 파일 수집 & 자연 정렬 -----
paths = glob.glob(os.path.join(nii_dir, "betas_session*.nii.gz"))

def session_key(p):
    # betas_session01.nii.gz -> 1 (자연 정렬용)
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
assert "num" in stim.columns and "shared1000" in stim.columns, "subject2_stim_info.csv에 num/shared1000 컬럼이 없습니다."
# shared1000이 문자열일 수 있으니 bool로 안전 변환
shared_map = dict(zip(stim["num"].astype(int), stim["shared1000"].astype(str).str.lower().isin(["1","true","t","yes"])))

# ----- 73KID 순서대로 train/test 레이블 배열 만들기 -----
# cat[i] = 0(train/False) or 1(test/True)
cat = np.empty(N_total, dtype=np.int8)
for i, k in enumerate(kid_seq):
    v = shared_map.get(k, None)
    if v is None:
        raise ValueError(f"subject2_stim_info.csv에 num={k}가 없습니다.")
    cat[i] = 1 if v else 0

n_train = int((cat == 0).sum())
n_test  = int((cat == 1).sum())
print(f"[INFO] 분할 예측: train={n_train}, test={n_test}")

# (선언된 기대치와 다르면 여기서 경고만)
if n_train != 27000 or n_test != 3000:
    print(f"[WARN] 기대치(27000/3000)와 다름: train={n_train}, test={n_test}")

# ----- 첫 파일에서 공간 정보 가져오기 -----
first_img = nib.load(paths[0])
affine = first_img.affine
hdr = first_img.header.copy()
# 공간 크기/세션 첫 번째의 볼륨 수
first_shape = first_img.shape  # (X,Y,Z,T_sess)
X, Y, Z = first_shape[:3]
print(f"[INFO] 공간 크기: {(X,Y,Z)}")

# ----- 출력 NIfTI를 메모리 절약형으로 만들기 -----
# nibabel은 저장 시 전체 배열을 들고 있어야 하므로,
# 최종 배열을 디스크-백업 메모리맵으로 만들고 채운 뒤 그걸 dataobj로 넘겨 저장합니다.
train_mm = np.memmap(os.path.join(BASE, "._tmp_train_mm.dat"), mode="w+", dtype=np.float32, shape=(X, Y, Z, n_train))
test_mm  = np.memmap(os.path.join(BASE, "._tmp_test_mm.dat"),  mode="w+", dtype=np.float32, shape=(X, Y, Z, n_test))

train_ptr = 0
test_ptr = 0
global_ptr = 0

# ----- 세션 파일 순회하며 스트리밍 복사 -----
for p in paths:
    img = nib.load(p)
    # dataobj는 지연 로딩 지원 → 슬라이스 단위로 뽑아서 복사
    dataobj = img.dataobj  # shape: (X,Y,Z,T_local)
    assert dataobj.shape[:3] == (X, Y, Z), f"공간 크기 불일치: {p}"
    T_local = dataobj.shape[3] if dataobj.ndim == 4 else 1

    for t in range(T_local):
        label = cat[global_ptr]  # 0 or 1
        vol = np.asanyarray(dataobj[..., t], dtype=np.float32)  # (X,Y,Z)
        if label == 0:
            train_mm[..., train_ptr] = vol
            train_ptr += 1
        else:
            test_mm[..., test_ptr] = vol
            test_ptr += 1
        global_ptr += 1

# ----- 개수 검증 -----
assert global_ptr == N_total, f"총 볼륨 수 불일치: 채운={global_ptr}, 기대={N_total}"
assert train_ptr == n_train and test_ptr == n_test, f"train/test 채움 개수 불일치: train={train_ptr}/{n_train}, test={test_ptr}/{n_test}"

# ----- NIfTI 저장 -----
# 헤더 dtype을 float32로 명시(필요 시 변경)
hdr.set_data_dtype(np.float32)

img_train = nib.Nifti1Image(train_mm, affine, header=hdr)
img_test  = nib.Nifti1Image(test_mm,  affine, header=hdr)

nib.save(img_train, out_train)
nib.save(img_test,  out_test)

print("[DONE]")
print(f"Saved: {out_train}  shape={(X,Y,Z,n_train)}")
print(f"Saved: {out_test}   shape={(X,Y,Z,n_test)}")
