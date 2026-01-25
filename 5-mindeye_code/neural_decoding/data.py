import os
import re
import glob
import random
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from torch import nn
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import nibabel as nib

import utils

class TrainDataset(Dataset): # ses단위로 실행
    def __init__(self, fmri_path, tsv_path, image_path, mask_path, transform, train=1):
        self.fmri_path = fmri_path
        self.tsv_path = tsv_path
        self.image_path = image_path
        self.mask_path = mask_path
        self.train = train  # 'train' or 'test'
        self.transform = transform # PIL.Image -> tensor

        # train index 뽑아두기
        df = pd.read_csv(self.tsv_path, sep='\t')
        self.valid_indices = df[df['train'] == train].index.tolist()

        # mask처리 - sub마다 maskload(shape: (Z, Y, X))
        # sub = re.search(r"(sub-\d+)", self.fmri_path).group(1)
        # mask_file = os.path.join(self.mask_path, f"{sub}_nsdgeneral.nii.gz")
        # mask_data = nib.load(mask_file).get_fdata()
        # mask_bool = (mask_data == 1)  # mask에 해당하는 부분 true 
        # self.mask_tensor = torch.tensor(mask_bool).nonzero(as_tuple=True) # tensor로 변환 + 위치로 저장 -> 속도 빠름

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx): 
        actual_idx = self.valid_indices[idx]  # idx -> 기존 데이터에서의 진짜 인덱스

        # 해당 볼륨만 로드 (4D → 3D)
        # fmri = nib.load(self.fmri_path).dataobj
        # fmri_vol = torch.tensor(fmri[:, :, :, actual_idx]).float()

        # 마스크 적용: (Z, Y, X) → (N,)
        # fmri_vol = fmri_vol[self.mask_tensor]

        # npy 로딩
        fmri_all = np.load(self.fmri_path, mmap_mode='r', allow_pickle=True)  # shape: (T, N_voxels)
        fmri_vol = torch.tensor(fmri_all[actual_idx]).float()  # shape: (N_voxels,)

        # image column 한 행만 로딩
        row = pd.read_csv(self.tsv_path, sep='\t', skiprows=range(1, actual_idx + 1), nrows=1)
        image_id = row['image'].values[0]

        # 이미지 로딩
        image_path = os.path.join(self.image_path, image_id + '.jpg')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_vol, image

class TestDataset(Dataset): # sub단위로 실행
    def __init__(self, fmri_info_list, image_path, low_image_path, mask_path, transform, use_low_image):
        self.fmri_info_list = fmri_info_list  # 리스트: {'image_id': str, 'fmri_volumes': [(path1, idx1), (path2, idx2), (path3, idx3)]}
        self.image_path = image_path
        self.low_image_path = low_image_path
        self.mask_path = mask_path
        self.transform = transform # PIL.Image -> tensor
        self.use_low_image = use_low_image

        # mask처리 - sub마다 maskload(shape: (Z, Y, X))
        # sub = re.search(r"(sub-\d+)", self.fmri_info_list[0]['fmri_volumes'][0][0]).group(1)
        # mask_file = os.path.join(self.mask_path, f"{sub}_nsdgeneral.nii.gz")
        # mask_data = nib.load(mask_file).get_fdata()
        # mask_bool = (mask_data == 1)  # mask에 해당하는 부분 true 
        # self.mask_tensor = torch.tensor(mask_bool).nonzero(as_tuple=True) # tensor로 변환 + 위치로 저장 -> 속도 빠름

        
    def __len__(self):
        return len(self.fmri_info_list)

    def __getitem__(self, idx):
        info = self.fmri_info_list[idx]
        image_id = info['image_id']
        fmri_list = info['fmri_volumes']  # [(path1, idx1), (path2, idx2), (path3, idx3)]
        
        fmri_vols = []
        for path, i in fmri_list:
            data = np.load(path, mmap_mode='r', allow_pickle=True)  # shape: (T, N_voxels)
            fmri_vol = torch.tensor(data[i]).float()  # shape: (N_voxels,)
            fmri_vols.append(fmri_vol)


            # data = nib.load(path).dataobj
            # fmri_vol = torch.tensor(data[:, :, :, i]).float()
            # # 마스크 적용: (Z, Y, X) → (N,)
            # fmri_vol = fmri_vol[self.mask_tensor]  
            # fmri_vols.append(fmri_vol)

        # idx당 하나의 volume 생성
        fmri_avg = torch.stack(fmri_vols).mean(0)  # mean(0): voxel-wise 평균 -> 결과 shape(X, Y, Z)

        image_path = os.path.join(self.image_path, image_id + '.jpg')
        low_image_path = os.path.join(self.low_image_path, image_id + ".jpg")

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        
        if self.use_low_image:
            low_image = Image.open(low_image_path).convert('RGB')
            if self.transform:
                low_image = self.transform(low_image)
        else:
            low_image = []

        return fmri_avg, image, low_image, image_id
    
class hug_TrainDataset(Dataset): # ses단위로 실행
    def __init__(self, beta_path, stimuli_path, image_path, transform):
        # 기존 NPZ 방식 (주석 처리)
        # self.data = np.load(fmri_path, mmap_mode='r', allow_pickle=True) # 포인터만 받아와서 메모리에 올라온 것은 아님
        # self.fmri = self.data['beta']
        # self.cocoid = self.data['stimuli']

        # 새로운 NPY 방식
        self.fmri = np.load(beta_path, mmap_mode='r', allow_pickle=True)  # shape: (N, voxels)
        self.cocoid = np.load(stimuli_path, allow_pickle=True)  # shape: (N,)
        self.image_path = image_path
        self.transform = transform # PIL.Image -> tensor


    def __len__(self):
        return len(self.cocoid)

    def __getitem__(self, idx):
        # fMRI 데이터 로딩
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        image_path = os.path.join(self.image_path, self.cocoid[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_vol, image

class hug_TestDataset(Dataset): # ses단위로 실행
    def __init__(self, beta_path, stimuli_path, image_path, low_image_path, transform, use_low_image):
        # 기존 NPZ 방식 (주석 처리)
        # self.data = np.load(fmri_path, mmap_mode='r', allow_pickle=True) # 포인터만 받아와서 메모리에 올라온 것은 아님
        # self.fmri = self.data['beta']
        # self.cocoid = self.data['stimuli']

        # 새로운 NPY 방식
        self.fmri = np.load(beta_path, mmap_mode='r', allow_pickle=True)  # shape: (N, voxels)
        self.cocoid = np.load(stimuli_path, allow_pickle=True)  # shape: (N,)
        self.image_path = image_path
        self.low_image_path = low_image_path
        self.transform = transform # PIL.Image -> tensor
        self.use_low_image = use_low_image

    def __len__(self):
        return len(self.cocoid)

    def __getitem__(self, idx):
        # fMRI 데이터 로딩
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        image_path = os.path.join(self.image_path, self.cocoid[idx])
        low_image_path = os.path.join(self.low_image_path, self.cocoid[idx])

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.use_low_image:
            low_image = Image.open(low_image_path).convert('RGB')
            if self.transform:
                low_image = self.transform(low_image)
        else:
            low_image = []

        return fmri_vol, image, low_image, self.cocoid[idx]
    
class FuncSpatial_TrainDataset(Dataset): # ses단위로 실행
    def __init__(self, fmri_path, image_path, transform):
        self.data = np.load(fmri_path, mmap_mode='r', allow_pickle=True) # 포인터만 받아와서 메모리에 올라온 것은 아님
        self.fmri = self.data['X']
        self.cocoid = self.data['Y']
        self.image_path = image_path
        self.transform = transform # PIL.Image -> tensor

        self.seq_len = self.fmri.shape[1]
       

    def __len__(self):
        return len(self.cocoid)

    def __getitem__(self, idx): 
        # fMRI 데이터 로딩
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        image_path = os.path.join(self.image_path, self.cocoid[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_vol, image

class FuncSpatial_TestDataset(Dataset): # ses단위로 실행
    def __init__(self, fmri_path, image_path, low_image_path, transform, use_low_image):
        self.data = np.load(fmri_path, mmap_mode='r', allow_pickle=True) # 포인터만 받아와서 메모리에 올라온 것은 아님
        self.fmri = self.data['X']
        self.cocoid = self.data['Y']
        self.image_path = image_path
        self.low_image_path = low_image_path
        self.transform = transform # PIL.Image -> tensor
        self.use_low_image = use_low_image

        self.seq_len = self.fmri.shape[1]

    def __len__(self):
        return len(self.cocoid)

    def __getitem__(self, idx): 
        # fMRI 데이터 로딩
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        image_path = os.path.join(self.image_path, self.cocoid[idx])
        low_image_path = os.path.join(self.low_image_path, self.cocoid[idx])

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.use_low_image:
            low_image = Image.open(low_image_path).convert('RGB')
            if self.transform:
                low_image = self.transform(low_image)
        else:
            low_image = []

        return fmri_vol, image, low_image, self.cocoid[idx]
    
class hug2_TrainDataset(Dataset): # ses단위로 실행
    def __init__(self, fmri_path, image_path, transform):
        self.data = np.load(fmri_path, mmap_mode='r', allow_pickle=True) # 포인터만 받아와서 메모리에 올라온 것은 아님
        self.fmri = self.data['X']
        self.cocoid = self.data['Y']
        self.image_path = image_path
        self.transform = transform # PIL.Image -> tensor
       

    def __len__(self):
        return len(self.cocoid)

    def __getitem__(self, idx): 
        # fMRI 데이터 로딩
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        image_path = os.path.join(self.image_path, self.cocoid[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_vol, image

class hug2_TestDataset(Dataset): # ses단위로 실행
    def __init__(self, fmri_path, image_path, transform):
        self.data = np.load(fmri_path, mmap_mode='r', allow_pickle=True) # 포인터만 받아와서 메모리에 올라온 것은 아님
        self.fmri = self.data['X']
        self.cocoid = self.data['Y']
        self.image_path = image_path
        self.transform = transform # PIL.Image -> tensor

    def __len__(self):
        return len(self.cocoid)

    def __getitem__(self, idx): 
        # fMRI 데이터 로딩
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        image_path = os.path.join(self.image_path, self.cocoid[idx])

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_vol, image, self.cocoid[idx]

# DataLoader를 한 번만 사용하기 위해 Dataset을 묶어줌
class MultiSubjectDatasetStrict(Dataset):
    '''
        {'subj01': Dataset[idx], 'subj02': Dataset[idx], ...}
    '''
    def __init__(self, datasets: dict):
        self.datasets = datasets
        self.subjs = list(datasets.keys())
        lengths = [len(datasets[subj]) for subj in self.subjs]
        assert len(set(lengths)) == 1, f"모든 subject 길이가 같아야 합니다: {lengths}"
        self._len = lengths[0]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # {subj: (fmri, image)} 반환
        return {subj: self.datasets[subj][idx] for subj in self.subjs}

def sub1_train_dataset(args): # ses단위로 실행

    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    transform = transforms.ToTensor()
    
    # 세션 자동 추출
    # pattern = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-*/func/sub-01_ses-*_desc-betaroizscore.nii.gz"
    pattern = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-*/func/sub-01_ses-*_desc-betaroizscore.npy"
    fmri_files = glob.glob(pattern)
    sessions = sorted([re.search(r'ses-(\d+)', f).group(1) for f in fmri_files]) # ex) 01,02 ... 추출

    train_datasets = []

    for ses in sessions:
        # fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-{ses}/func/sub-01_ses-{ses}_desc-betaroizscore.nii.gz"
        fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-{ses}/func/sub-01_ses-{ses}_desc-betaroizscore.npy"
        tsv_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-{ses}/func/sub-01_ses-{ses}_task-image_events.tsv"
        image_path = f"{root_dir}/{image_dir}"
        mask_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01"
        
        train_datasets.append(TrainDataset(fmri_path, tsv_path, image_path, mask_path, transform, train=1))

    # Dataset 합침
    train_dataset = ConcatDataset(train_datasets)
    
    return train_dataset

def sub1_test_dataset(args): # sub단위로 실행

    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    code_dir = args.code_dir
    output_dir= args.output_dir
    transform = transforms.ToTensor()
    use_low_image = args.use_low_image

    # 모든 nii 경로 뽑음
    # pattern = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-*/func/sub-01_ses-*_desc-betaroizscore.nii.gz"
    pattern = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-*/func/sub-01_ses-*_desc-betaroizscore.npy"
    fmri_files = sorted(glob.glob(pattern))

    # 모든 trial 정리
    '''
    image_info = [
        {'image_id': 'img_001', 'fmri_path': 'ses-01.nii.gz', 'volume_idx': 4},
        {'image_id': 'img_001', 'fmri_path': 'ses-03.nii.gz', 'volume_idx': 12},
        {'image_id': 'img_001', 'fmri_path': 'ses-08.nii.gz', 'volume_idx': 7},
        {'image_id': 'img_002', 'fmri_path': 'ses-01.nii.gz', 'volume_idx': 9},
        {'image_id': 'img_002', 'fmri_path': 'ses-03.nii.gz', 'volume_idx': 14},
    ]
    '''
    image_info = []
    for fmri_path in fmri_files:
        # tsv_path = fmri_path.replace('_desc-betaroizscore.nii.gz', '_task-image_events.tsv')
        tsv_path = fmri_path.replace('_desc-betaroizscore.npy', '_task-image_events.tsv')
        df = pd.read_csv(tsv_path, sep='\t')
        test_df = df[df['train'] == 0].copy()

        for idx, row in test_df.iterrows():
            image_id = row['image']
            image_info.append({'image_id': image_id, 'fmri_path': fmri_path, 'volume_idx': idx})

    # image_info를 Dataframe으로 만들고 group처리
    image_df = pd.DataFrame(image_info)
    grouped = image_df.groupby('image_id')
    
    # 한 image에 해당하는 모든 fMRI idx정보 저장
    '''
    {
        'image_id': 'img_001',
        'fmri_volumes': [
            ('ses-01.nii.gz', 4),
            ('ses-03.nii.gz', 12),
            ('ses-08.nii.gz', 7)
        ]
    },
    '''
    averaged_list = []
    for image_id, group in grouped:
        fmri_volumes = [(row['fmri_path'], row['volume_idx']) for _, row in group.iterrows()]
        averaged_list.append({
            'image_id': image_id,
            'fmri_volumes': fmri_volumes
        })
    
    image_path = f"{root_dir}/{image_dir}"
    low_image_path = f"{root_dir}/{code_dir}/{output_dir}/low_recons"
    mask_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01"

    test_dataset = TestDataset(averaged_list, image_path, low_image_path, mask_path, transform, use_low_image)

    return test_dataset

def sub1_train_dataset_hug(args):
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    transform = transforms.ToTensor()

    # 기존 NPZ 방식 (주석 처리)
    # fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/sub-01_beta-train_nsdgeneral.npy"
    # train_dataset = hug_TrainDataset(fmri_path, image_path, transform)

    # 새로운 NPY 방식 - beta와 stimuli 파일 분리
    beta_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/sub-01_beta-train_nsdgeneral.npy"
    stimuli_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/sub-01_stimuli-train.npy"
    image_path = f"{root_dir}/{image_dir}"

    train_dataset = hug_TrainDataset(beta_path, stimuli_path, image_path, transform)

    return train_dataset

def sub1_test_dataset_hug(args):
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    code_dir = args.code_dir
    output_dir= args.output_dir
    transform = transforms.ToTensor()
    use_low_image = args.use_low_image

    # 기존 NPZ 방식 (주석 처리)
    # fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/sub-01_beta-test_nsdgeneral.npy"
    # test_dataset = hug_TestDataset(fmri_path, image_path, low_image_path, transform, use_low_image)

    # 새로운 NPY 방식 - beta와 stimuli 파일 분리
    beta_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/sub-01_beta-test_nsdgeneral.npy"
    stimuli_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/sub-01_stimuli-test.npy"
    image_path = f"{root_dir}/{image_dir}"
    low_image_path = f"{root_dir}/{code_dir}/{output_dir}/mindeye1_metric_ourdata/low_recons_test"

    test_dataset = hug_TestDataset(beta_path, stimuli_path, image_path, low_image_path, transform, use_low_image)

    return test_dataset

def sub1_train_dataset_FuncSpatial(args):
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    transform = transforms.ToTensor()
    
    fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/sub-01_fmri_with_labels_train.npz"
    image_path = f"{root_dir}/{image_dir}"
 
    train_dataset = FuncSpatial_TrainDataset(fmri_path, image_path, transform)
    
    return train_dataset

def sub1_test_dataset_FuncSpatial(args):
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    code_dir = args.code_dir
    output_dir= args.output_dir
    transform = transforms.ToTensor()
    use_low_image = args.use_low_image
    
    fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/sub-01_fmri_with_labels_test.npz"
    image_path = f"{root_dir}/{image_dir}"
    low_image_path = f"{root_dir}/{code_dir}/{output_dir}/low_recons"
 
    test_dataset = FuncSpatial_TestDataset(fmri_path, image_path, low_image_path, transform, use_low_image)
    
    return test_dataset

def train_dataset_hug2(args, subj_names):
    """
    subj_names: ['sub-01', 'sub-02', ...] 
    """
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir

    transform = transforms.ToTensor()

    datasets = {}
    for subj in subj_names:
        fmri_path = os.path.join(root_dir, fmri_dir, fmri_detail_dir, subj, f"{subj}_fmri_with_labels_train.npz")
        image_path = os.path.join(root_dir, image_dir)

        dataset = hug2_TrainDataset(fmri_path, image_path, transform)
        datasets[subj] = dataset

    return datasets  # {'subj01': Dataset, 'subj02': Dataset, ...}

def test_dataset_hug2(args, subj_names):
    """
    subj_names: ['sub-01', 'sub-02', ...] 
    """
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir

    transform = transforms.ToTensor()

    datasets = {}
    for subj in subj_names:
        fmri_path = os.path.join(root_dir, fmri_dir, fmri_detail_dir, subj, f"{subj}_fmri_with_labels_test.npz")
        image_path = os.path.join(root_dir, image_dir)

        dataset = hug2_TestDataset(fmri_path, image_path, transform)
        datasets[subj] = dataset

    return datasets  # {'subj01': Dataset, 'subj02': Dataset, ...}

def get_dataloader_hug2(args, subj_names):
    '''
    batch = {
        "sub-01": (
            torch.stack([fmri_01_idx0, fmri_01_idx1]),   # shape: [각 batch size, volumne개수]
            torch.stack([img_01_idx0, img_01_idx1])      # shape: [각 batch size, 3, 224, 224]
        ),
        "sub-02": (
            torch.stack([fmri_02_idx0, fmri_02_idx1]),   # shape: [각 batch size, volumne개수]
            torch.stack([img_02_idx0, img_02_idx1])      # shape: [각 batch size, 3, 224, 224]
        ),
        "sub-03": (
            torch.stack([fmri_03_idx0, fmri_03_idx1]),   # shape: [각 batch size, volumne개수]
            torch.stack([img_03_idx0, img_03_idx1])      # shape: [각 batch size, 3, 224, 224]
        ),
    }
        
    '''
    train_loaders={}
    inference_loaders={}

    if args.mode == 'train':
        train_datasets = train_dataset_hug2(args, subj_names)
        multi_train_dataset = MultiSubjectDatasetStrict(train_datasets) # {subj: (fmri, image)} 반환

        # 여기서 batch_size는 각각의 개수임. 합친 수 아님 ex) 각각 10 -> 전체 30
        train_loaders = DataLoader(multi_train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=True)
        return train_loaders
    
    if args.mode == 'inference':
        inference_datasets = test_dataset_hug2(args, subj_names)

        multi_inference_dataset = MultiSubjectDatasetStrict(inference_datasets) # {subj: (fmri, image)} 반환

        # 여기서 batch_size는 각각의 개수임. 합첸 수 아님 ex) 각각 10 -> 전체 30 
        # 다만 fine-tunning에서는 subject 하나만 사용하니까 batch size를 올려도 됨
        inference_loaders = DataLoader(multi_inference_dataset, batch_size=args.inference_batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=True)
        return inference_loaders


# mindeye1 & connectomind
def get_dataloader(args):

    # 제거할 index 집합
    # drop_idx = {0, 5, 8, 10, 15, 18} # low
    # drop_idx = {1,4,11,14} # high
 
    if args.mode == 'train':
        train_dataset = sub1_train_dataset_hug(args)
        # keep_idx = [i for i in range(train_dataset.seq_len) if i not in drop_idx]
        # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn_factory_train(keep_idx))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=True)
        return train_loader
    
    if args.mode == 'inference':
        test_dataset = sub1_test_dataset_hug(args)
        # keep_idx = [i for i in range(test_dataset.seq_len) if i not in drop_idx]
        # test_loader = DataLoader(test_dataset, batch_size=args.inference_batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=args.is_shuffle, worker_init_fn=worker_init_fn, collate_fn=collate_fn_factory_test(keep_idx))
        test_loader = DataLoader(test_dataset, batch_size=args.inference_batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=args.is_shuffle)
        return test_loader
    
    
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32  # worker별 고유 seed 생성
    np.random.seed(seed)
    random.seed(seed)    

def collate_fn_factory_train(keep_idx):
    """
    keep_idx 리스트를 클로저로 잡는 collate_fn 생성기
    """
    def collate_fn(batch):
        # batch: list of tuples [(fmri, label), ...]
        fmri_batch, label_batch = zip(*batch)
        fmri_batch = torch.stack(fmri_batch, dim=0)        # [B, 20, 2056]
        fmri_batch = fmri_batch[:, keep_idx, :]            # [B, len(keep_idx), 2056]
        label_batch = torch.stack(label_batch, dim=0)
        return fmri_batch, label_batch
    return collate_fn

def collate_fn_factory_test(keep_idx):
    """
    keep_idx 리스트를 클로저로 잡는 collate_fn 생성기
    """
    def collate_fn(batch):
        # batch: list of tuples [(fmri_vol, image, low_image, image_id), ...]
        fmri_list, image_list, low_list, id_list = zip(*batch)

        # 1) fmri: [B, seq_len, feats]
        fmri_batch = torch.stack(fmri_list, dim=0)
        # 필요한 ROI만 남기기 (keep_idx 는 미리 정의된 리스트)
        fmri_batch = fmri_batch[:, keep_idx, :]

        # 2) image: [B, C, H, W] (transform이 Tensor 변환까지 했을 경우)
        image_batch = torch.stack(image_list, dim=0)

        # 3) low_image: use_low_image 여부에 따라 텐서 혹은 빈 리스트
        if isinstance(low_list[0], torch.Tensor):
            low_batch = torch.stack(low_list, dim=0)
        else:
            # low_image가 [] 로 들어오는 경우, 그냥 빈 리스트 묶음으로 전달
            low_batch = list(low_list)

        # 4) image_id: 문자열 ID 리스트
        id_batch = list(id_list)

        # 최종 반환: fmri, image, low_image, id
        return fmri_batch, image_batch, low_batch, id_batch
    return collate_fn

    