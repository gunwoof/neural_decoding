import numpy as np

path = "/research/wsi/research/03-Neural_decoding/3-bids/2-derivatives/1-beta/beta_mni_2mm/sub-01/sub-01_beta_test_schaefer100.npz"

data = np.load(path, allow_pickle=True)
print("keys:", data.files)

for k in data.files:
    v = data[k]
    print(f"{k}: type={type(v)}, shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")
