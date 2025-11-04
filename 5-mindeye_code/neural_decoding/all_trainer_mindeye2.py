import re, os
import gc
import time
from tqdm import tqdm
import wandb

import numpy as np
from scipy import stats
from accelerate import Accelerator
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import transforms

from utils import img_augment_high, mixup, mixco_nce_loss, cosine_anneal, soft_clip_loss, topk, batchwise_cosine_similarity, log_gradient_norms, check_nan_and_log, reconstruction, get_unique_path, img_augment_low, soft_cont_loss, unclip_recon, sdxl_recon, save_gt_vs_recon_images_extended

def pre_train(args, subj_names, train_data, models, optimizer, lr_scheduler):

    device = args.device
    num_epochs = args.num_epochs
    mixup_pct = args.mixup_pct
    prior_loss_coefficient = args.prior_loss_coefficient
    nce_loss_coefficient = args.nce_loss_coefficient
    lowlevel_loss_coefficient = args.lowlevel_loss_coefficient

    scaler = GradScaler(enabled=False) # autocast scaler 인스턴스 생성
    subj_names = subj_names


    # model 정의
    clip_extractor = models["clip"]
    mindeye2 = models["mindeye2"]
    vae = models["vae"]
    cnx = models["cnx"]
    l1 = models["l1"]
    optimizer = optimizer
    lr_scheduler = lr_scheduler

    # log list
    losses, lrs = [], []
    progress_bar = tqdm(range(0, num_epochs), ncols=50)
    for epoch in progress_bar:
        mindeye2.train()

        # 기본 log
        loss_prior_sum = 0.0  # prior loss의 누적합 -> 평균 구할 때 쓰임
        loss_nce_sum = 0.0 # Negative Contrastive Estimation loss의 누적합 -> 평균 구할 때 쓰임
        loss_blurry_sum = 0.0 # l1 loss의 누적합 -> 평균 구할 때 쓰임
        loss_blurry_cont_sum = 0.0 # contloss의 누적합 -> 평균 구할 때 쓰임

        
        for index, batch in enumerate(train_data): # enumerate: index와 값을 같이 반환
            # global step 계산
            global_step = epoch * len(train_data) + index

            optimizer.zero_grad()

            fmri_list, image_list = {}, {}
            perm_list, betas_list, select_list = {}, {}, {}
            use_mix = epoch < int(mixup_pct * num_epochs)

            for subj in subj_names:
                fmri_vol, img = batch[subj]          
                fmri_vol = fmri_vol.to(device, non_blocking=True)
                img = img.to(device,  non_blocking=True)
                
                # epoch의 1/3 지점 까지만 mixup 사용
                if use_mix:
                    fmri_vol, perm, betas, select = mixup(fmri_vol)
                    perm_list[subj] = perm
                    betas_list[subj] = betas
                    select_list[subj] = select

                fmri_list[subj] = fmri_vol
                image_list[subj] = img

            image = torch.cat([image_list[subj] for subj in subj_names], dim=0)      
            if use_mix:
                perm   = torch.cat([perm_list[subj] for subj in subj_names], dim=0)
                betas  = torch.cat([betas_list[subj] for subj in subj_names], dim=0)
                select = torch.cat([select_list[subj] for subj in subj_names], dim=0)    

            # 고해상도 이미지 증강
            image = img_augment_high(image)

            with autocast(dtype=torch.bfloat16):
                #### forward 계산 + loss 계산 ####
                with torch.no_grad():
                    # target 정의
                    clip_target = clip_extractor(image)[0]

                # Shared-subject latent space(each -> 4096)
                voxel_ridge_list = [mindeye2.ridge(fmri_list[subj], i) for i, subj in enumerate(subj_names)]
                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

                # Residual MLP backbone 
                voxel_backbone, voxel_retrieval, voxel_lowlevels = mindeye2.backbone(voxel_ridge)

                # forward(Diffusion prior) -> prior loss
                loss_prior, _ = mindeye2.diffusion_prior(text_embed=voxel_backbone, image_embed=clip_target)
                
                # forward(retrieval submodule) -> contrstive loss(mixco_nce_loss + soft_clip_loss)
                clip_voxels_norm = nn.functional.normalize(voxel_retrieval.flatten(1), dim=-1).float()
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1).float()
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

                with torch.no_grad():
                    image_enc = vae.encode(2*image-1).latent_dist.mode() * 0.18215
                loss_blurry = l1(image_enc_pred, image_enc)

                with torch.no_grad():
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

            # torch.cuda.empty_cache() # gpu 메모리 cache삭제
            # gc.collect() # # gpu 메모리 안 쓰는거 삭제

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
                "epoch": epoch,
                "train/num_steps": index + 1,  # 현재 iteration
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
            progress_bar.set_postfix(**logs) # cli에 시각화
            wandb.log(logs, step=global_step) # wandb에 시각화

        if epoch >= 140 and epoch % 5 == 0:
        # if epoch %/10 == 0:
            save_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, "mindeye2_metric", f"mindeye2_pretrain_{epoch}_{args.experiment_name}.pt")
            save_path = get_unique_path(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 경로 없으면 생성
            torch.save(mindeye2.state_dict(), save_path)

        torch.cuda.empty_cache() # gpu 메모리 cache삭제
        gc.collect() # # gpu 메모리 안 쓰는거 삭제

def pre_train_continous(args, subj_names, train_data, models, optimizer, lr_scheduler):

    device = args.device
    num_epochs = args.num_epochs
    mixup_pct = args.mixup_pct
    prior_loss_coefficient = args.prior_loss_coefficient
    nce_loss_coefficient = args.nce_loss_coefficient
    lowlevel_loss_coefficient = args.lowlevel_loss_coefficient

    scaler = GradScaler(enabled=False) # autocast scaler 인스턴스 생성
    subj_names = subj_names


    # model 정의
    clip_extractor = models["clip"]
    mindeye2 = models["mindeye2"]
    vae = models["vae"]
    cnx = models["cnx"]
    l1 = models["l1"]
    optimizer = optimizer
    lr_scheduler = lr_scheduler

    ckpt_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, "mindeye2_metric", "mindeye2_pretrain_170_1257.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    mindeye2.load_state_dict(checkpoint, strict=False)

    # log list
    losses, lrs = [], []
    progress_bar = tqdm(range(0, num_epochs), ncols=50)
    for epoch in progress_bar:
        mindeye2.train()

        # 기본 log
        loss_prior_sum = 0.0  # prior loss의 누적합 -> 평균 구할 때 쓰임
        loss_nce_sum = 0.0 # Negative Contrastive Estimation loss의 누적합 -> 평균 구할 때 쓰임
        loss_blurry_sum = 0.0 # l1 loss의 누적합 -> 평균 구할 때 쓰임
        loss_blurry_cont_sum = 0.0 # contloss의 누적합 -> 평균 구할 때 쓰임

        
        for index, batch in enumerate(train_data): # enumerate: index와 값을 같이 반환
            # global step 계산
            global_step = epoch * len(train_data) + index

            optimizer.zero_grad()

            fmri_list, image_list = {}, {}
            perm_list, betas_list, select_list = {}, {}, {}
            use_mix = epoch < int(mixup_pct * num_epochs)

            for subj in subj_names:
                fmri_vol, img = batch[subj]          
                fmri_vol = fmri_vol.to(device, non_blocking=True)
                img = img.to(device,  non_blocking=True)
                
                # epoch의 1/3 지점 까지만 mixup 사용
                if use_mix:
                    fmri_vol, perm, betas, select = mixup(fmri_vol)
                    perm_list[subj] = perm
                    betas_list[subj] = betas
                    select_list[subj] = select

                fmri_list[subj] = fmri_vol
                image_list[subj] = img

            image = torch.cat([image_list[subj] for subj in subj_names], dim=0)      
            if use_mix:
                perm   = torch.cat([perm_list[subj] for subj in subj_names], dim=0)
                betas  = torch.cat([betas_list[subj] for subj in subj_names], dim=0)
                select = torch.cat([select_list[subj] for subj in subj_names], dim=0)    

            # 고해상도 이미지 증강
            image = img_augment_high(image)

            with autocast(dtype=torch.bfloat16):
                #### forward 계산 + loss 계산 ####
                with torch.no_grad():
                    # target 정의
                    clip_target = clip_extractor(image)[0]

                # Shared-subject latent space(each -> 4096)
                voxel_ridge_list = [mindeye2.ridge(fmri_list[subj], i) for i, subj in enumerate(subj_names)]
                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

                # Residual MLP backbone 
                voxel_backbone, voxel_retrieval, voxel_lowlevels = mindeye2.backbone(voxel_ridge)

                # forward(Diffusion prior) -> prior loss
                loss_prior, _ = mindeye2.diffusion_prior(text_embed=voxel_backbone, image_embed=clip_target)
                
                # forward(retrieval submodule) -> contrstive loss(mixco_nce_loss + soft_clip_loss)
                clip_voxels_norm = nn.functional.normalize(voxel_retrieval.flatten(1), dim=-1).float()
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1).float()
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

                with torch.no_grad():
                    image_enc = vae.encode(2*image-1).latent_dist.mode() * 0.18215
                loss_blurry = l1(image_enc_pred, image_enc)

                with torch.no_grad():
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

            # torch.cuda.empty_cache() # gpu 메모리 cache삭제
            # gc.collect() # # gpu 메모리 안 쓰는거 삭제

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
                "epoch": epoch,
                "train/num_steps": index + 1,  # 현재 iteration
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
            progress_bar.set_postfix(**logs) # cli에 시각화
            wandb.log(logs, step=global_step) # wandb에 시각화

        if epoch >= 130 and epoch % 5 == 0:
        # if epoch %/10 == 0:
            save_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, "mindeye2_metric", f"mindeye2_pretrain_continous_{epoch}_{args.experiment_name}.pt")
            save_path = get_unique_path(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 경로 없으면 생성
            torch.save(mindeye2.state_dict(), save_path)

        torch.cuda.empty_cache() # gpu 메모리 cache삭제
        gc.collect() # # gpu 메모리 안 쓰는거 삭제



def fine_tunning_train(args, subj_names, train_data, models, optimizer, lr_scheduler):

    # train argument
    device = args.device
    experiment_name = args.experiment_name
    num_epochs = args.num_epochs
    mixup_pct = args.mixup_pct
    prior_loss_coefficient = args.prior_loss_coefficient
    nce_loss_coefficient = args.nce_loss_coefficient
    lowlevel_loss_coefficient = args.lowlevel_loss_coefficient
    subj_names = subj_names

    # test argument
    seed = args.seed

    scaler = GradScaler(enabled=False) # autocast scaler 인스턴스 생성
    subj_names = subj_names

    # model 정의
    clip_extractor = models["clip"]
    mindeye2 = models["mindeye2"]
    vae = models["vae"]
    cnx = models["cnx"]
    l1 = models["l1"]
    optimizer = optimizer
    lr_scheduler = lr_scheduler
 
  
    losses, lrs = [], [] # log list
    progress_bar = tqdm(range(0, num_epochs), ncols=50)
    for epoch in progress_bar:
        
        # 기본 log
        loss_prior_sum = 0.0  # prior loss의 누적합 -> 평균 구할 때 쓰임
        loss_nce_sum = 0.0 # Negative Contrastive Estimation loss의 누적합 -> 평균 구할 때 쓰임
        loss_blurry_sum = 0.0 # l1 loss의 누적합 -> 평균 구할 때 쓰임
        loss_blurry_cont_sum = 0.0 # contloss의 누적합 -> 평균 구할 때 쓰임

        #### train #### 
        mindeye2.train()
        for index, batch in enumerate(train_data): # enumerate: index와 값을 같이 반환
            # global step 계산
            global_step = epoch * len(train_data) + index

            optimizer.zero_grad()

            fmri_list, image_list = {}, {}
            perm_list, betas_list, select_list = {}, {}, {}
            use_mix = epoch < int(mixup_pct * num_epochs)

            for subj in subj_names:
                fmri_vol, img = batch[subj]          
                fmri_vol = fmri_vol.to(device, non_blocking=True)
                img = img.to(device,  non_blocking=True)
                
                # epoch의 1/3 지점 까지만 mixup 사용
                if use_mix:
                    fmri_vol, perm, betas, select = mixup(fmri_vol)
                    perm_list[subj] = perm
                    betas_list[subj] = betas
                    select_list[subj] = select

                fmri_list[subj] = fmri_vol
                image_list[subj] = img

            image = torch.cat([image_list[subj] for subj in subj_names], dim=0)      
            if use_mix:
                perm   = torch.cat([perm_list[subj] for subj in subj_names], dim=0)
                betas  = torch.cat([betas_list[subj] for subj in subj_names], dim=0)
                select = torch.cat([select_list[subj] for subj in subj_names], dim=0)    

            # 고해상도 이미지 증강
            image = img_augment_high(image)

            with autocast(dtype=torch.bfloat16):
                #### forward 계산 + loss 계산 ####
                with torch.no_grad():
                    # target 정의
                    clip_target = clip_extractor(image)[0]

                # Shared-subject latent space(each -> 4096)
                voxel_ridge_list = [mindeye2.ridge(fmri_list[subj], i) for i, subj in enumerate(subj_names)]
                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

                # Residual MLP backbone 
                voxel_backbone, voxel_retrieval, voxel_lowlevels = mindeye2.backbone(voxel_ridge)

                # forward(Diffusion prior) -> prior loss
                loss_prior, _ = mindeye2.diffusion_prior(text_embed=voxel_backbone, image_embed=clip_target)
                
                # forward(retrieval submodule) -> contrstive loss(mixco_nce_loss + soft_clip_loss)
                clip_voxels_norm = nn.functional.normalize(voxel_retrieval.flatten(1), dim=-1).float()
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1).float()
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

                with torch.no_grad():
                    image_enc = vae.encode(2*image-1).latent_dist.mode() * 0.18215
                loss_blurry = l1(image_enc_pred, image_enc)

                with torch.no_grad():
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
                "epoch": epoch,
                "train/num_steps": index + 1,  # 현재 iteration
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
            progress_bar.set_postfix(**logs) # cli에 시각화
            wandb.log(logs, step=global_step) # wandb에 시각화

        if epoch >= 130 and epoch % 5 == 0:
        # if epoch % 10 == 0:
            save_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, "mindeye2_metric", f"mindeye2_finetunning_{epoch}_{experiment_name}.pt")
            save_path = get_unique_path(save_path)
            torch.save(mindeye2.state_dict(), save_path)

        torch.cuda.empty_cache() # gpu 메모리 cache삭제
        gc.collect() # # gpu 메모리 안 쓰는거 삭제



def inference_evaluate(args, subj_names, test_data, models, metrics, ckpt_dir):

    # train argument
    device = args.device
    experiment_name = args.experiment_name
    num_epochs = args.num_epochs
    subj_names = subj_names

    # test argument
    seed = args.seed
    scaler = GradScaler(enabled=False) # autocast scaler 인스턴스 생성

    # model 정의
    mindeye2 = models["mindeye2"]
    vae = models["vae"]
    clip_linear = models["clip_linear"]
    clip_text_model = models["clip_text_model"]
    token_to_text = models["token_to_text"]
    sdxl_unclip = models["sdxl_unclip"]
    base_text_embedder1 = models["base_text_embedder1"]
    base_text_embedder2 = models["base_text_embedder2"]
    sdxl = models["sdxl"]
    noise_scheduler = models["noise_scheduler"]

    # checkpoint 불러오기
    ckpt_files = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt") and "mindeye2_pretrain_continous" in f], key=lambda x: int(re.search(r"_(\d+)_\d+\.pt", os.path.basename(x)).group(1)) if re.search(r"_(\d+)_\d+\.pt", os.path.basename(x)) else 0, reverse=True)
    for ckpt_path in ckpt_files:
        print(f"Loading checkpoint from {ckpt_path}")

        # 파일명에서 숫자 인덱스 추출
        ckpt_name = os.path.basename(ckpt_path)
        match = re.search(r"_(\d+)_", ckpt_name)
        ckpt_num = int(match.group(1))

        #### inference ####
        all_targets = []
        all_targets_ids = []
        all_recons = []
        all_blurryrecons = []
        all_captions = []
        all_enhanced_recons = []
        all_final_recons = []

        mindeye2.eval()
        mindeye2.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        # 나머지 subject(2,5,7) weight 삭제 (메모리 절약)
        # del mindeye2.ridge.linears[1:]
        print(f"[디버그] ridge 내부 linears 개수: {len(mindeye2.ridge.linears)}")
        print(f"[디버그] ridge 내부 layer 구조:")
        print(f"  {[f'linears.{i}: ({layer.in_features} → {layer.out_features}, bias={layer.bias is not None})' for i, layer in enumerate(mindeye2.ridge.linears)]}")

        progress_bar = tqdm(enumerate(test_data), ncols=120)
        for index, batch in progress_bar:
            with torch.inference_mode():
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                
                # sub-01만 있음
                fmri_list = {}
                for subj in subj_names:
                    fmri_vol, img, image_id = batch[subj]          
                    fmri_vol = fmri_vol.to(device, non_blocking=True)
                    fmri_list[subj] = fmri_vol

                    # image와 image_id는 dict 형태로 담아두기
                    for i, img_id in zip(img, image_id):
                        tgt = transforms.Resize((256, 256), antialias=True)(i.cpu()).float()
                        all_targets.append(tgt)
                        all_targets_ids.append(img_id)
                       

                # Shared-subject latent space(each -> 4096)
                voxel_ridge_list = [mindeye2.ridge(fmri_list[subj], i) for i, subj in enumerate(subj_names)]
                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

                # Residual MLP backbone(4096 -> 256*1664)
                voxel_backbone, voxel_retrieval, voxel_lowlevels = mindeye2.backbone(voxel_ridge)

                # Diffusion prior(256*1664)
                loss_prior = mindeye2.diffusion_prior.p_sample_loop(voxel_backbone.shape, text_cond = dict(text_embed = voxel_backbone), timesteps = 20 , cond_scale = 1.)

                # SDXL unCLIP
                template = {
                    "jpg": torch.randn(args.inference_batch_size, 3, 1, 1).to(device),             # (B,3,1,1)
                    "original_size_as_tuple": torch.ones(args.inference_batch_size, 2).to(device) * 768,  # (B,2)
                    "crop_coords_top_left": torch.zeros(args.inference_batch_size, 2).to(device)          # (B,2)
                }
                out = sdxl_unclip.conditioner(template)
                vector_suffix = out["vector"][:, :1024].to(device) # vector(전역조건), crossattn(앞 부분)이 필요 없어서 1024-dim부분만 사용

                samples = unclip_recon(loss_prior, sdxl_unclip, vector_suffix, num_samples=1, device=device)

                # caption linear + caption 생성
                pred_caption_emb = clip_linear(loss_prior)
                generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
                generated_caption = token_to_text.batch_decode(generated_ids, skip_special_tokens=True)
                
                # low-level submodule
                image_enc_pred, transformer_feats = voxel_lowlevels
                blurred_images = (vae.decode(image_enc_pred/0.18215).sample/ 2 + 0.5).clamp(0,1)
                
                # SDXL unCLIP + caption + low-level submodule을 모아서 최종 재구성
                template2 = {
                    "txt": [""] * args.inference_batch_size,             
                    "original_size_as_tuple": torch.ones(args.inference_batch_size, 2).to(device) * 768,  # (B,2)
                    "crop_coords_top_left": torch.zeros(args.inference_batch_size, 2).to(device),  
                    "target_size_as_tuple": torch.zeros(args.inference_batch_size, 2).to(device) * 1024
                }
                out2 = sdxl.conditioner(template2)
                crossattn_c = out2["crossattn"].to(device) # cfg할 때 사용
                vector_c = out2["vector"][:,-1536:].to(device) # cfg할 때 사용

                negative_prompt = (
                    "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, "
                    "deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, "
                    "skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, "
                    "missing lips, ugly face, distorted face, extra legs, anime"
                )
                templete2_uc = {
                    "txt": [negative_prompt] * args.inference_batch_size,           
                    "original_size_as_tuple": torch.ones(args.inference_batch_size, 2).to(device) * 768,  # (B,2)
                    "crop_coords_top_left": torch.zeros(args.inference_batch_size, 2).to(device),  
                    "target_size_as_tuple": torch.zeros(args.inference_batch_size, 2).to(device) * 1024
                }
                out2_uc = sdxl.conditioner(templete2_uc)
                crossattn_uc = out2_uc["crossattn"].to(device) # cfg할 때 사용
                vector_uc = out2_uc["vector"].to(device) # cfg할 때 사용

                enhanced_samples = sdxl_recon(args.inference_batch_size, samples, generated_caption, sdxl, base_text_embedder1, base_text_embedder2, vector_c, crossattn_uc, vector_uc, num_samples=1, img2img_timepoint=13, device=device)
                final_recons = enhanced_samples*.75 + blurred_images*.25

                for b in range(samples.shape[0]):
                    all_recons.append(transforms.Resize((256,256), antialias=True)(samples[b].cpu()).float())
                    all_blurryrecons.append(transforms.Resize((256,256), antialias=True)(blurred_images[b].cpu()).float())
                    all_captions.append(generated_caption[b])
                    all_enhanced_recons.append(transforms.Resize((256,256), antialias=True)(enhanced_samples[b].cpu()).float())
                    all_final_recons.append(transforms.Resize((256,256), antialias=True)(final_recons[b].cpu()).clamp(0,1).float())

                torch.cuda.empty_cache() # gpu 메모리 cache삭제
                gc.collect() # gpu 메모리 안 쓰는거 삭제

        all_recons = torch.stack(all_recons, dim=0)  # [N, 3, H, W]
        all_blurryrecons = torch.stack(all_blurryrecons, dim=0)  # [N, 3, H, W]
        all_enhanced_recons = torch.stack(all_enhanced_recons, dim=0)  # [N, 3, H, W]
        all_final_recons = torch.stack(all_final_recons, dim=0)  # [N, 3, H, W]
        all_targets = torch.stack(all_targets, dim=0)  # [3, H, W]여러개 -> [N, 3, H, W]

        # 🔍 디버그 출력
        print("\n========== Debug: Tensor Shapes ==========")
        print(f"all_recons shape         : {tuple(all_recons.shape)}")
        print(f"all_blurryrecons shape   : {tuple(all_blurryrecons.shape)}")
        print(f"all_enhanced_recons shape: {tuple(all_enhanced_recons.shape)}")
        print(f"all_final_recons shape   : {tuple(all_final_recons.shape)}")
        print(f"all_targets shape        : {tuple(all_targets.shape)}")
        # 혹시 채널 수나 dtype이 안 맞을 수도 있으니 추가로 확인
        print("\n========== Debug: Tensor Details ==========")
        print(f"all_final_recons dtype: {all_final_recons.dtype}, range=({all_final_recons.min():.3f}, {all_final_recons.max():.3f})")
        print(f"all_targets dtype     : {all_targets.dtype}, range=({all_targets.min():.3f}, {all_targets.max():.3f})")
        print("===========================================\n")   

        #### evaluate ####
        results = {}

        compare_sets = {
            "final_recons": all_final_recons,
            "enhanced_recons": all_enhanced_recons,
            "recons": all_recons,
            "blurryrecons": all_blurryrecons
        }

        for name, recons in compare_sets.items():
            # 각 비교 세트별 metric 계산
            sub_results = {}

            # PixCorr / SSIM
            sub_results["PixCorr"] = metrics["pixcorr"](recons, all_targets)
            sub_results["SSIM"] = metrics["ssim"](recons, all_targets)

            # AlexNet
            sub_results["AlexNet_2"] = metrics["alexnet2"]["metric_fn"](
                args, recons, all_targets,
                metrics["alexnet2"]["model"],
                metrics["alexnet2"]["preprocess"],
                metrics["alexnet2"]["layer"]
            )
            sub_results["AlexNet_5"] = metrics["alexnet5"]["metric_fn"](
                args, recons, all_targets,
                metrics["alexnet5"]["model"],
                metrics["alexnet5"]["preprocess"],
                metrics["alexnet5"]["layer"]
            )

            # CLIP / Inception / EfficientNet / SwAV
            sub_results["CLIP"] = metrics["clip"]["metric_fn"](
                args, recons, all_targets,
                metrics["clip"]["model"],
                metrics["clip"]["preprocess"]
            )
            sub_results["Inception"] = metrics["inception"]["metric_fn"](
                args, recons, all_targets,
                metrics["inception"]["model"],
                metrics["inception"]["preprocess"]
            )
            sub_results["EfficientNet_B1"] = metrics["efficientnet"]["metric_fn"](
                args, recons, all_targets,
                metrics["efficientnet"]["model"],
                metrics["efficientnet"]["preprocess"]
            )
            sub_results["SwAV"] = metrics["swav"]["metric_fn"](
                args, recons, all_targets,
                metrics["swav"]["model"],
                metrics["swav"]["preprocess"]
            )

            results[name] = sub_results

            print(f"\n===== {name} vs Target =====")
            for metric_name, score in sub_results.items():
                print(f"{metric_name:15}: {score:.4f}")
            print("=" * 40)

        # wandb에 기록 (평가 세트별로 구분)
        for recon_name, sub_results in results.items():
            wandb.log(
                {f"eval/{recon_name}/epoch{ckpt_num}_{k}": v for k, v in sub_results.items()},
                step=ckpt_num
            )

        # CLIP 점수 기준 (final_recons 기준)
        current_score = results["final_recons"].get("CLIP", 0.0)
        if current_score > 0.7:

            # save_recons 저장
            recons_dir = os.path.join(args.root_dir, args.code_dir, args.output_dir, "mindeye2_metric", "recon_benchmark")
            save_gt_vs_recon_images_extended(all_targets, all_recons, all_blurryrecons, all_enhanced_recons, all_final_recons, all_targets_ids, save_dir=recons_dir, layout='horizontal')

            # 결과를 텍스트 파일로 저장
            result_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, "mindeye2_metric", f"mindeye2_finetunning_metrics_{ckpt_num}_{experiment_name}.txt")
            result_path = get_unique_path(result_path)
            with open(result_path, "w") as f:
                for recon_name, sub_results in results.items():
                    f.write(f"==== {recon_name} vs Target ====\n")
                    for metric_name, score in sub_results.items():
                        f.write(f"{metric_name}: {score:.4f}\n")
                    f.write("\n")

        torch.cuda.empty_cache() # gpu 메모리 cache삭제
        gc.collect() # gpu 메모리 안 쓰는거 삭제







                