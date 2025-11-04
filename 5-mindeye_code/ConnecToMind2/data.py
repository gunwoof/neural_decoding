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

class TrainDataset_ourdata(Dataset): # ses단위로 실행
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

class TestDataset_ourdata(Dataset): # ses단위로 실행
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
    

def train_dataset(args):
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    subjects = args.subjects
    transform = transforms.ToTensor()

    datasets = []
    for sub in subjects:
        fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/{sub}/{sub}_fmri_with_labels_train.npz"
        image_path = f"{root_dir}/{image_dir}"
        datasets.append(TrainDataset_ourdata(fmri_path, image_path, transform))

    train_dataset = ConcatDataset(datasets)
    return train_dataset

def test_dataset(args):
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    subjects = args.subjects
    transform = transforms.ToTensor()

    datasets = []
    for sub in subjects:
        fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/{sub}/{sub}_fmri_with_labels_test.npz"
        image_path = f"{root_dir}/{image_dir}"
        datasets.append(TestDataset_ourdata(fmri_path, image_path, transform))

    test_dataset = ConcatDataset(datasets)
    return test_dataset

def get_dataloader(args):
    '''
        train_loader.shape 
            fmri_vol.shape  [batch_size, roir개수, (voxel개수+padding)],
            image.shape  [batch_size, 3, 224, 224]
        test_loader.shape  
            fmri_vol.shape  [inference_batch_size, roir개수, (voxel개수+padding)],
            image.shape  [inference_batch_size, 3, 224, 224]
            image_id.shape  [inference_batch_size,]  ex) ['coco2017_14.jpg' 'coco2017_14.jpg', ...]
    '''
    # 제거할 index 집합
    # drop_idx = {0, 5, 8, 10, 15, 18} # low
    # drop_idx = {1,4,11,14} # high
 
    if args.mode == 'train':
        train_dataset = train_dataset(args)
        # keep_idx = [i for i in range(train_dataset.seq_len) if i not in drop_idx]
        # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn_factory_train(keep_idx))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=True)
        return train_loader
    
    if args.mode == 'inference':
        test_dataset = test_dataset(args)
        # keep_idx = [i for i in range(test_dataset.seq_len) if i not in drop_idx]
        # test_loader = DataLoader(test_dataset, batch_size=args.inference_batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=args.is_shuffle, worker_init_fn=worker_init_fn, collate_fn=collate_fn_factory_test(keep_idx))
        test_loader = DataLoader(test_dataset, batch_size=args.inference_batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=False)
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