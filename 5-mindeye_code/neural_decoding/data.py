import os
import re
import glob
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import nibabel as nib

import utils
from args import parse_args


class TrainDataset(Dataset):
    def __init__(self, fmri_path, tsv_path, image_path, transform, train=1):
        self.fmri_path = fmri_path
        self.tsv_path = tsv_path
        self.image_path = image_path
        self.train = train  # 'train' or 'test'
        self.transform = transform

        # train & test 각각 index 뽑아두기
        df = pd.read_csv(self.tsv_path, sep='\t')
        self.valid_indices = df[df['train'] == train].index.tolist()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx): 
        actual_idx = self.valid_indices[idx]  # idx -> 기존 데이터에서의 진짜 인덱스

        # 해당 볼륨만 로드 (4D → 3D)
        fmri = nib.load(self.fmri_path).get_fdata()
        fmri_vol = torch.tensor(fmri[:, :, :, actual_idx]).float()

        # image column 한 행만 로딩
        row = pd.read_csv(self.tsv_path, sep='\t', skiprows=range(1, actual_idx + 1), nrows=1)
        image_id = row['image'].values[0]

        # 이미지 로딩
        image_path = os.path.join(self.image_path, image_id + '.jpg')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_vol, image

class TestDataset(Dataset):
    def __init__(self, fmri_info_list, image_dir, transform):
        self.fmri_info_list = fmri_info_list  # 리스트: {'image_id': str, 'fmri_volumes': [(path1, idx1), (path2, idx2), (path3, idx3)]}
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.fmri_info_list)

    def __getitem__(self, idx):
        info = self.fmri_info_list[idx]
        image_id = info['image_id']
        fmri_list = info['fmri_volumes']  # [(path1, idx1), (path2, idx2), (path3, idx3)]
        
        fmri_vols = []
        for path, i in fmri_list:
            data = nib.load(path).get_fdata()
            fmri_vol = torch.tensor(data[:, :, :, i]).float()
            fmri_vols.append(fmri_vol)

        # idx당 하나의 volume 생성
        fmri_avg = torch.stack(fmri_vols).mean(0)  # mean(0): voxel-wise 평균 -> 결과 shape(X, Y, Z)

        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_avg, image

def sub1_train_dataset():
    args = parse_args()

    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    transform = transforms.ToTensor()
    
    # 세션 자동 추출
    pattern = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-*/func/sub-01_ses-*_desc-betaroizscore.nii.gz"
    fmri_files = glob.glob(pattern)
    sessions = sorted([re.search(r'ses-(\d+)', f).group(1) for f in fmri_files]) # ex) 01,02 ... 추출

    train_datasets = []

    for ses in sessions:
        fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-{ses}/func/sub-01_ses-{ses}_desc-betaroizscore.nii.gz"
        tsv_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-{ses}/func/sub-01_ses-{ses}_task-image_events.tsv"
        image_path = f"{root_dir}/{image_dir}"
        
        train_datasets.append(TrainDataset(fmri_path, tsv_path, image_path, transform, train=1))

    # Dataset 합침
    train_dataset = ConcatDataset(train_datasets)
    
    return train_dataset

def sub1_test_dataset():
    args = parse_args()

    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    transform = transforms.ToTensor()

    # 모든 nii 경로 뽑음
    pattern = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-*/func/sub-01_ses-*_desc-betaroizscore.nii.gz"
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
        tsv_path = fmri_path.replace('_desc-betaroizscore.nii.gz', '_task-image_events.tsv')
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
    
    test_dataset = TestDataset(averaged_list, os.path.join(root_dir, image_dir), transform)

    return test_dataset

def get_dataloaders():
    args = parse_args()

    # 시드 고정
    utils.seed_everything(args.seed) 
    
    if args.mode == 'train':
        train_dataset = sub1_train_dataset()
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=args.is_shuffle)
        return train_loader
    
    if args.mode == 'test':
        test_dataset = sub1_test_dataset()
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)
        return test_loader
    
    

    