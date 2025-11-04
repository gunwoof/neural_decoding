#!/usr/bin/env python3
import argparse, glob, os, itertools
import numpy as np
import pandas as pd
from math import acos
from numpy.linalg import norm
import ants

def load_affine(path):
    import numpy as np
    # 1) 먼저 텍스트 4x4 시도
    try:
        M = np.loadtxt(path)
        if M.shape == (4, 4):
            return M
    except Exception:
        pass

    # 2) ANTs/ITK 변환 읽기 (binary/ITK text)
    import ants
    tx = ants.read_transform(path)  # ANTsTransform

    # antspyx에서는 affine의 parameters가 보통 12개:
    # [a11 a12 a13 a21 a22 a23 a31 a32 a33 tx ty tz]  (row-major)
    p = np.array(tx.parameters, dtype=float).ravel()
    if p.size < 12:
        raise ValueError(f"{path}: unexpected parameter length {p.size}")

    A = p[:9].reshape(3, 3)          # 3x3 affine linear part
    t = p[9:12]                      # translation (ITK parameter)
    # fixedParameters는 보통 center of rotation (cx, cy, cz)
    fp = np.array(getattr(tx, "fixedParameters", []), dtype=float).ravel()
    c = fp[:3] if fp.size >= 3 else np.zeros(3)

    # ITK의 AffineTransform는 center가 있을 수 있으므로,
    # 최종 오프셋 b = t + A @ c - c  를 사용해야 정확한 4x4가 됩니다.
    b = t + A @ c - c

    M = np.eye(4, dtype=float)
    M[:3, :3] = A
    M[:3,  3] = b
    return M


def rot_trans_scale(M):
    R = M[:3,:3]
    t = M[:3, 3]
    # column-norm 기반 scale (단순 스케일; shear와 분리되진 않음)
    s = norm(R, axis=0)
    Rn = R / s
    return Rn, t, s

def rotation_angle_deg(Ra, Rb):
    # 상대 회전행렬의 회전각
    Rrel = Rb.T @ Ra
    tr = np.clip((np.trace(Rrel) - 1)/2, -1.0, 1.0)
    return np.degrees(acos(tr))

def points_for_delta():
    # 원점 주변 3D 포인트 9개(원점, 축방향 ±100, 대각선 두 점)
    P = np.array([
        [0,0,0],
        [100,0,0],[-100,0,0],
        [0,100,0],[0,-100,0],
        [0,0,100],[0,0,-100],
        [100,100,100],[-100,-100,-100],
    ], dtype=float)
    return P

def apply_affine(M, pts):
    pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])
    out = (M @ pts_h.T).T[:, :3]
    return out

def compare_two(Ma, Mb, a_name, b_name):
    # 1) 단순 행렬 차이
    frob = norm(Ma - Mb, 'fro')

    # 2) R/t/s 분해 후 차이
    Ra, ta, sa = rot_trans_scale(Ma)
    Rb, tb, sb = rot_trans_scale(Mb)
    rot_deg = rotation_angle_deg(Ra, Rb)
    trans_mm = norm(ta - tb)
    scale_l2 = norm(sa - sb)

    # 3) 좌표 변환 결과 차이(포인트 세트)
    P = points_for_delta()
    Pa = apply_affine(Ma, P)
    Pb = apply_affine(Mb, P)
    dists = norm(Pa - Pb, axis=1)
    mean_pt = float(np.mean(dists))
    max_pt  = float(np.max(dists))

    return {
        "A": a_name,
        "B": b_name,
        "frob_dist": float(frob),
        "rot_diff_deg": float(rot_deg),
        "trans_diff_mm": float(trans_mm),
        "scale_diff_l2": float(scale_l2),
        "mean_point_delta": mean_pt,
        "max_point_delta": max_pt,
    }

def main():
    ap = argparse.ArgumentParser(description="Affine(.mat) 다중 비교")
    ap.add_argument("--glob", required=True, help="예: '/path/sub-01_ses-*_t1_to_beta_0GenericAffine.mat'")
    ap.add_argument("--base", help="기준 파일(지정 시 base vs others 비교)")
    ap.add_argument("--pairwise", action="store_true", help="모두-대-모두 비교")
    ap.add_argument("--out", default="affine_compare.csv", help="결과 CSV 경로")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit("매칭되는 .mat 파일이 없습니다.")

    results = []
    mats = {f: load_affine(f) for f in files}

    if args.base:
        if args.base not in mats:
            # base가 글롭에 안 걸렸다면 따로 로드 시도
            if os.path.exists(args.base):
                mats[args.base] = load_affine(args.base)
            else:
                raise SystemExit(f"--base 파일을 찾을 수 없습니다: {args.base}")
        base = args.base
        for f in files:
            if f == base: 
                continue
            results.append(compare_two(mats[base], mats[f], os.path.basename(base), os.path.basename(f)))

    if args.pairwise:
        for a, b in itertools.combinations(files, 2):
            results.append(compare_two(mats[a], mats[b], os.path.basename(a), os.path.basename(b)))

    if not args.base and not args.pairwise:
        # 기본: 첫 번째를 기준으로 base vs others
        base = files[0]
        for f in files[1:]:
            results.append(compare_two(mats[base], mats[f], os.path.basename(base), os.path.basename(f)))

    df = pd.DataFrame(results)
    if len(df):
        df.to_csv(args.out, index=False)
        print(f"[OK] {len(df)}개 비교 결과를 '{args.out}'에 저장했습니다.")
        # 간단 요약
        print(df.describe().to_string())
    else:
        print("비교 결과가 없습니다(입력 옵션 확인).")

if __name__ == "__main__":
    main()
