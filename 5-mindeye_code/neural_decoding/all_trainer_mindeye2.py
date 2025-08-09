import os
import gc
import time
from tqdm import tqdm
import wandb

import numpy as np
from scipy import stats
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity

from utils import img_augment_high, mixup, mixco_nce_loss, cosine_anneal, soft_clip_loss, topk, batchwise_cosine_similarity, log_gradient_norms, check_nan_and_log, reconstruction, get_unique_path, img_augment_low, soft_cont_loss

def pre_train(args, subj_names, data, models, optimizer, lr_scheduler):

    device = args.device
    num_epochs = args.num_epochs
    mixup_pct = args.mixup_pct
    prior_loss_coefficient = args.prior_loss_coefficient
    nce_loss_coefficient = args.nce_loss_coefficient
    lowlevel_loss_coefficient = args.lowlevel_loss_coefficient

    scaler = GradScaler() # autocast scaler 인스턴스 생성
    
    subj_names = subj_names

    # model 정의
    clip_extractor = models["clip"]
    mindeye2 = models["mindeye2"]
    vae = models["vae"]
    noise_scheduler = models["noise_scheduler"]
    cnx = models["cnx"]
    l1 = models["l1"]
    optimizer = optimizer
    lr_scheduler = lr_scheduler

    # log list
    losses, lrs = [], []

    progress_bar = tqdm(range(0, num_epochs), ncols=1200)
    for epoch in progress_bar:
        mindeye2.train()

        # 기본 log
        loss_prior_sum = 0.0  # prior loss의 누적합 -> 평균 구할 때 쓰임
        loss_nce_sum = 0.0 # Negative Contrastive Estimation loss의 누적합 -> 평균 구할 때 쓰임
        loss_blurry_sum = 0.0 # l1 loss의 누적합 -> 평균 구할 때 쓰임
        loss_blurry_cont_sum = 0.0 # contloss의 누적합 -> 평균 구할 때 쓰임

        voxel_iters, image_iters = {}, {}, {}, {}
        perm_iters, betas_iters, select_iters = {}, {}, {}
        
        for subj, each_data in data.items(): # subject를 묶어서 dic으로 반환
            for index, (fmri_vol, image) in enumerate(subj[each_data]): # 각 subject
                # data gpu 올리기 - amp 사용
                fmri_vol = fmri_vol  
                image = image   
                # epoch의 1/3 지점 까지만 mixup 사용
                if epoch < int(mixup_pct * num_epochs):
                    fmri_vol, perm, betas, select = mixup(fmri_vol)

                    perm_iters[f"{subj}_{index}"] = perm
                    betas_iters[f"{subj}_{index}"] = betas
                    select_iters[f"{subj}_{index}"] = select

                voxel_iters[f"{subj}_{index}"] = fmri_vol
                image_iters[f"{subj}_{index}"] = image

        count_iteration = sum(1 for k in voxel_iters if k.startswith("subj01_"))
        for i in range(count_iteration): # enumerate: index와 값을 같이 반환
            with autocast():
                # global step 계산
                global_step = epoch * count_iteration + index

                # gradient 초기화
                optimizer.zero_grad() 

                fmri = [voxel_iters[f"{subj}_{i}"].to(device) for subj in subj_names] # fMRI -> GPU
                image = [image_iters[f"{subj}_{i}"].to(device) for subj in subj_names] # Image -> GPU

                # image augmentation
                image = img_augment_high(image)

                #### forward 계산 + loss 계산 ####
                # target 정의
                clip_target = clip_extractor(image).float()

                # epoch의 1/3 지점 까지만 mixup 사용
                if epoch < int(mixup_pct * num_epochs):
                    perm_list = [perm_iters[f"{subj}_{i}"].to(device) for s in subj_names]
                    perm = torch.cat(perm_list, dim=0)
                    betas_list = [betas_iters[f"{subj}_{i}"].to(device) for s in subj_names]
                    betas = torch.cat(betas_list, dim=0)
                    select_list = [select_iters[f"{subj}_{i}"].to(device) for s in subj_names]
                    select = torch.cat(select_list, dim=0)

                # Shared-subject latent space(each -> 4096)
                voxel_ridge_list = [mindeye2.ridge(fmri[i], i) for i in range(len(subj_names))]
                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)
                
                # Residual MLP backbone 
                voxel_backbone, voxel_retrieval, voxel_lowlevels = mindeye2.backbone(voxel_ridge)

                # forward(Diffusion prior) -> prior loss
                loss_prior, prior_out = mindeye2.diffusion_prior(text_embed=voxel_backbone, image_embed=clip_target)
                
                # forward(retrieval submodule) -> contrstive loss(mixco_nce_loss + soft_clip_loss)
                clip_voxels_norm = nn.functional.normalize(voxel_retrieval.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                # mixco_nce_loss(1/3) + soft_loss_temps(2/3)
                if epoch < int(mixup_pct * num_epochs):
                    nce_loss = mixco_nce_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006, 
                        perm=perm, betas=betas, select=select)
                else:
                    soft_loss_temps = cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    nce_loss = soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)
                
                # forward(low-level submodule) -> l1_loss + cnx_loss
                image_enc_pred, transformer_feats = voxel_lowlevels

                image_enc = vae.encode(2*image-1).latent_dist.mode() * 0.18215
                loss_blurry = l1(image_enc_pred, image_enc)

                mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1) # imagenet의 mean
                std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1) # imagenet의 std
                image_norm = (image - mean)/std
                image_aug = (img_augment_low(image) - mean)/std
                _, cnx_embeds = cnx(image_norm)
                _, cnx_aug_embeds = cnx(image_aug)
                cont_loss = soft_cont_loss(
                    nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    temp=0.2
                )

                # 최종 loss 정의
                loss = (prior_loss_coefficient * loss_prior) + (nce_loss_coefficient * nce_loss) + (lowlevel_loss_coefficient * (loss_blurry + 0.1*cont_loss)) 

                # NaN 체크 + 넘김
                if check_nan_and_log(global_step=index, fmri_vol=fmri_vol, voxel_backbone=voxel_backbone, voxel_retrieval=voxel_retrieval, voxel_lowlevels=voxel_lowlevels, loss=loss):
                    continue

                #### backward 계산 + update ####
                # gradient 계산 - amp사용
                scaler.scale(loss).backward() # amp사용

                # optimizer update - amp사용
                scaler.step(optimizer) # amp사용
                scaler.update() # amp사용
                torch.cuda.empty_cache() # gpu 메모리 cache삭제
                gc.collect() # # gpu 메모리 안 쓰는거 삭제

                # learning rate schedule update
                lr_scheduler.step()

                #### log ####
                # loss, lr 담아두기
                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                # loss 누적합
                loss_prior_sum += loss_prior.item()
                loss_nce_sum += nce_loss.item()
                loss_blurry_sum += loss_blurry.item()
                loss_blurry_cont_sum += cont_loss.item()

                logs = {
                    # 기본 학습 상태
                    "train/num_steps": i + 1,  # 현재 iteration
                    "train/lr": lrs[-1],
                    "train/global_step": global_step,
                    "train/epoch": epoch,
                    "train/loss": losses[-1],
                    "train/loss_nce": nce_loss.item(),
                    "train/loss_prior": loss_prior.item(),
                    "train/loss_blurry": loss_blurry.item(),
                    "train/loss_cont": cont_loss.item(),

                    # 디버그: fmri_vol
                    "debug/fmri_nan": float(torch.isnan(fmri_vol).any().item()),
                    "debug/fmri_min": fmri_vol.min().item(),
                    "debug/fmri_max": fmri_vol.max().item(),

                    # 디버그: voxel_backbone
                    "debug/voxel_backbone_nan": float(torch.isnan(voxel_backbone).any().item()),
                    "debug/voxel_backbone_min": voxel_backbone.min().item(),
                    "debug/voxel_backbone_max": voxel_backbone.max().item(),

                    # 디버그: voxel_retrieval
                    "debug/voxel_retrieval_nan": float(torch.isnan(voxel_retrieval).any().item()),
                    "debug/voxel_retrieval_min": voxel_retrieval.min().item(),
                    "debug/voxel_retrieval_max": voxel_retrieval.max().item(),

                    # 디버그: voxel_lowlevels[0] (image_enc_pred)
                    "debug/voxel_lowlevels_pred_nan": float(torch.isnan(voxel_lowlevels[0]).any().item()),
                    "debug/voxel_lowlevels_pred_min": voxel_lowlevels[0].min().item(),
                    "debug/voxel_lowlevels_pred_max": voxel_lowlevels[0].max().item(),

                    # 디버그: loss 값 NaN 여부
                    "debug/loss_nan": float(torch.isnan(loss).item()),
                }
                # 각 subject별 ridge 출력 log 추가
                for si, subj in enumerate(subj_names):
                    ridge_out = voxel_ridge_list[si]
                    logs[f"debug/{subj}_ridge_nan"] = float(torch.isnan(ridge_out).any().item())
                    logs[f"debug/{subj}_ridge_min"] = ridge_out.min().item()
                    logs[f"debug/{subj}_ridge_max"] = ridge_out.max().item()

                progress_bar.set_postfix(**logs) # cli에 시각화
                wandb.log(logs, step=global_step) # wandb에 시각화

    return mindeye2