"""
ConnecToMind2 Trainer (Model2 - New Architecture)

Epoch 로직:
    - Epoch 0: train + evaluation + metric (초기 성능 확인)
    - Epoch 1~249: train only
    - Epoch 250+: train + (5 단위로 evaluation + metric)

각 단계 설명:
    1. Train: fMRI [B, 200, input_dim] -> ConnecToMind2 -> FIR loss + FIM loss + L1 loss + ConvNext loss
       - FIR Loss: fMRI embedding vs CLIP embedding (MSE) - Q-T 마스크 적용
       - FIM Loss: Cross entropy (matching) - 마스크 없음
       - L1 Loss: VAE latent 예측
       - ConvNext Loss: feature contrastive
    2. Evaluation: fMRI -> ConnecToMind2 -> Versatile Diffusion -> Reconstructed Image
       - fmri_proj [B, 257, 768]: Versatile Diffusion의 image_embeds
       - lowlevel_l1 [B, 4, 28, 28]: VAE decode -> blurry image
    3. Metric: PixCorr, SSIM, AlexNet2/5, CLIP, Inception, EfficientNet, SwAV
"""

import os
import gc
from tqdm import tqdm
import wandb

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from utils import soft_cont_loss, img_augment, versatile_diffusion_reconstruct, save_recon_batch, load_recons_from_disk

# ============================================================================
# Main Training Loop
# ============================================================================

def train_evaluate_metric(args, train_loader, test_loaders, models, optimizer, lr_scheduler, metrics):
    """
    전체 학습 루프

    Epoch 로직:
        - Epoch 0: train + evaluation + metric
          -> 초기 모델 성능 확인 (baseline)

        - Epoch 1~249: train only
          -> 빠른 학습 진행

        - Epoch 250+: train + (5 단위로 evaluation + metric)
          -> epoch 250, 255, 260, ... 에서 평가
          -> 모델이 수렴 단계에서 성능 모니터링
    """
    device = args.device

    # ============ Models ============
    model = models["connectomind2"].to(device)
    versatile_diffusion = models["versatile_diffusion"]  # Pipeline은 to(device) 별도 처리
    if versatile_diffusion is not None:
        versatile_diffusion = versatile_diffusion.to(device)
    vae = models["vae"].to(device)
    # cnx = models["cnx"].to(device)  # NOTE: cnx 비활성화 (메모리 절약)

    # AMP scaler
    scaler = GradScaler()

    # Output directory
    output_dir = os.path.join(args.root_dir, "5-mindeye_code/ConnecToMind2", args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    global_step = 0

    print(f"\n{'='*60}")
    print(f"Starting Training: {args.experiment_name}")
    print(f"Total Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(args.num_epochs):

        # ============ Train ============
        print(f"\n[Epoch {epoch}] Training...")
        global_step, avg_loss = train_one_epoch(
            args, model, vae, train_loader,
            optimizer, lr_scheduler, scaler,
            epoch, global_step
        )
        print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")

        # ============ Evaluation + Metric ============
        # Epoch 0 또는 Epoch 250 이상에서 5 단위
        should_evaluate = (epoch == 0) or (epoch >= 250 and epoch % 5 == 0)
        # should_evaluate = (epoch >= 250 and epoch % 5 == 0)

        if should_evaluate:
            print(f"\n[Epoch {epoch}] Evaluation & Metrics...")

            # 이미지 저장 디렉토리
            recon_dir = os.path.join(output_dir, f"recons_epoch{epoch}")

            # Subject별 evaluation 및 metric 계산
            all_results = {}

            # ============ Step 1: 모든 Subject의 이미지 생성 및 저장 ============
            for sub, sub_loader in test_loaders.items():
                print(f"\n  [{sub}] Generating reconstructions...")
                num_samples = evaluate(args, model, vae, versatile_diffusion, sub_loader, epoch, sub, recon_dir)
                print(f"  [{sub}] Saved {num_samples} images to disk")

            # ============ Step 2: 디스크에서 로드하여 Metric 계산 ============
            print(f"\n  Computing metrics from saved images...")
            for sub in args.subjects:
                print(f"\n  [{sub}] Loading images and computing metrics...")

                # 디스크에서 이미지 로드
                sub_recons, sub_targets = load_recons_from_disk(recon_dir, sub)

                # Metric 계산
                sub_results = compute_metrics(args, sub_recons, sub_targets, metrics)
                all_results[sub] = sub_results

                # Subject별 결과 출력
                print(f"  [{sub}] Results:")
                for name, score in sub_results.items():
                    print(f"    {name:15}: {score:.4f}")

                # 메모리 정리
                del sub_recons, sub_targets
                torch.cuda.empty_cache()
                gc.collect()

            # Subject 평균 계산
            results = {}
            metric_names = list(all_results[args.subjects[0]].keys())
            for name in metric_names:
                scores = [all_results[sub][name] for sub in args.subjects]
                results[name] = np.mean(scores)

            # 평균 결과 출력
            print(f"\n{'='*40}")
            print(f"Epoch {epoch} Average Metrics (across {len(args.subjects)} subjects):")
            print(f"{'='*40}")
            for name, score in results.items():
                print(f"  {name:15}: {score:.4f}")
            print(f"{'='*40}\n")

            # WandB logging (평균 + subject별)
            if args.wandb_log:
                wandb.log({f"eval/{k}": v for k, v in results.items()}, step=global_step)
                for sub in args.subjects:
                    wandb.log({f"eval/{sub}/{k}": v for k, v in all_results[sub].items()}, step=global_step)

            # ============ Model Saving (매 evaluation마다 저장) ============
            # 모델 저장
            model_path = os.path.join(output_dir, f"{args.model_name}_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': results,
                'metrics_per_subject': all_results,
            }, model_path)
            print(f"Model saved: {model_path}")
            print(f"Reconstructions saved: {recon_dir}")

            # Metric 결과 텍스트 저장
            metric_path = os.path.join(output_dir, f"metrics_epoch{epoch}.txt")
            with open(metric_path, "w") as f:
                f.write(f"Epoch: {epoch}\n")
                f.write(f"{'='*40}\n")
                f.write("Average Metrics:\n")
                for name, score in results.items():
                    f.write(f"  {name}: {score:.4f}\n")
                f.write(f"\n{'='*40}\n")
                f.write("Per-Subject Metrics:\n")
                for sub in args.subjects:
                    f.write(f"\n  [{sub}]\n")
                    for name, score in all_results[sub].items():
                        f.write(f"    {name}: {score:.4f}\n")
            print(f"Metrics saved: {metric_path}")

            # 메모리 정리
            del all_results
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}\n")

    return model


# ============================================================================
# Train Function
# ============================================================================

def train_one_epoch(args, model, vae, train_loader, optimizer, lr_scheduler, scaler, epoch, global_step):
    """
    한 epoch 학습 (Model2 - New Architecture)

    단계:
        1. fMRI -> ConnecToMind2 forward
           - fmri [B, 200, input_dim] -> Region-level embedding -> [B, 200, 768]
           - Connectome-Q-Former -> [B, 201, 768]
           - Output Projection -> fmri_proj [B, 257, 768]
           - Low-level Decoder -> lowlevel_l1 [B, 4, 28, 28], lowlevel_aux [B, 49, 512]
           - loss_fir: FIR loss (MSE, Q-T 마스크 적용)
           - loss_fim: FIM loss (BCE, 마스크 없음)

        2. Low-level L1 loss 계산
           - target: vae.encode(image) -> [B, 4, 28, 28]
           - pred: lowlevel_l1

        3. ConvNext contrastive loss 계산
           - target: cnx(image) -> [B, 49, 512]
           - pred: lowlevel_aux

        4. Total loss = fir_weight * loss_fir
                      + fim_weight * loss_fim
                      + lowlevel_weight * (loss_l1 + 0.1 * loss_cnx)
    """
    device = args.device
    model.train()

    losses_log = []
    lrs_log = []

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch}", ncols=150)

    for batch_idx, (fmri, image) in progress_bar:
        # Global step 계산
        current_step = global_step + batch_idx

        # Gradient 초기화
        optimizer.zero_grad()

        # Data -> GPU
        fmri = fmri.to(device, non_blocking=True)      # [B, 200, input_dim]
        image = image.to(device, non_blocking=True)    # [B, 3, 224, 224]

        with autocast():
            # ============ Forward ============
            outputs = model(fmri, image, device)

            # Q-Former losses (FIR + FIM)
            loss_fir = outputs["loss_fir"]
            loss_fim = outputs["loss_fim"]

            # ============ Low-level L1 Loss ============
            # VAE encode target
            with torch.no_grad():
                vae_target = vae.encode(2 * image - 1).latent_dist.mode() * 0.18215  # [B, 4, 28, 28]

            loss_l1 = F.l1_loss(outputs["lowlevel_l1"], vae_target)

            # ============ ConvNext Contrastive Loss (MindEye2 방식) ============
            # NOTE: cnx 비활성화 - gradient 폭발 문제
            # with torch.no_grad():
            #     # ImageNet normalization
            #     mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            #     std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

            #     # MindEye2: 원본 이미지 사용 (resize 없음)
            #     image_norm = (image - mean) / std
            #     image_aug = (img_augment(image) - mean) / std

            #     _, cnx_feats = cnx(image_norm)       # [B, HW, 512]
            #     _, cnx_aug_feats = cnx(image_aug)    # [B, HW, 512]

            # loss_cnx = soft_cont_loss(
            #     F.normalize(outputs["lowlevel_aux"].reshape(-1, outputs["lowlevel_aux"].shape[-1]), dim=-1),
            #     F.normalize(cnx_feats.reshape(-1, cnx_feats.shape[-1]), dim=-1),
            #     F.normalize(cnx_aug_feats.reshape(-1, cnx_aug_feats.shape[-1]), dim=-1),
            #     temp=0.125  # MindEye2 원본 temp
            # )
            loss_cnx = torch.tensor(0.0, device=device)

            # ============ Total Loss (MindEye2 방식) ============
            loss = (args.fir_weight * loss_fir
                    + args.fim_weight * loss_fim
                    + args.lowlevel_weight * (loss_l1/0.18215))

        # ============ Backward ============
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()

        # Learning rate scheduler step
        lr_scheduler.step()

        # ============ Logging ============
        losses_log.append(loss.item())
        lrs_log.append(optimizer.param_groups[0]['lr'])

        logs = {
            "train/epoch": epoch,
            "train/step": current_step,
            "train/loss": loss.item(),
            "train/loss_fir": loss_fir.item(),
            "train/loss_fim": loss_fim.item(),
            "train/loss_l1": loss_l1.item(),
            "train/loss_cnx": loss_cnx.item(),
            "train/lr": lrs_log[-1],
            "train/grad_norm": grad_norm.item(),  # Gradient norm 로깅
        }

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            fir=f"{loss_fir.item():.4f}",
            fim=f"{loss_fim.item():.4f}",
            l1=f"{loss_l1.item():.4f}",
            cnx=f"{loss_cnx.item():.4f}"
        )

        if args.wandb_log:
            wandb.log(logs, step=current_step)

    # GPU 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()

    return global_step + len(train_loader), np.mean(losses_log)


# ============================================================================
# Evaluation Function (Reconstruction)
# ============================================================================

@torch.no_grad()
def evaluate(args, model, vae, versatile_diffusion, test_loader, epoch, subject, save_dir):
    """
    Evaluation: fMRI -> 이미지 reconstruction (Model2 - Versatile Diffusion)
    배치마다 디스크에 저장하여 메모리 사용량 최소화

    단계:
        1. fMRI -> ConnecToMind2.inference()
           - fmri_proj: [B, 257, 768] - Versatile Diffusion의 image_embeds (conditioning)
           - lowlevel_l1: [B, 4, 28, 28] - VAE latent

        2. VAE decode lowlevel_l1 -> blurry image [B, 3, 224, 224]
           - 512x512로 resize

        3. Versatile Diffusion (image variation)
           - image: blurry image (low-level structure)
           - image_embeds: fmri_proj [B, 257, 768] (high-level semantic conditioning)
           -> Reconstructed image [B, 3, H, W]

        4. 배치마다 디스크에 저장 (Subject별 디렉토리)

    Args:
        subject: Subject ID (예: 'sub-01')
        save_dir: 이미지 저장 디렉토리

    Returns:
        num_samples: 처리한 샘플 수
    """
    device = args.device
    model.eval()

    # Generator for reproducibility
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    num_samples = 0

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader),
                        desc=f"Evaluation Epoch {epoch} [{subject}]", ncols=120)

    for batch_idx, (fmri, gt_images, image_ids) in progress_bar:
        # Data -> GPU
        fmri = fmri.to(device)

        # ============ Forward Inference ============
        outputs = model.inference(fmri)
        fmri_proj = outputs["fmri_proj"]      # [B, 257, 768] - conditioning
        lowlevel_l1 = outputs["lowlevel_l1"]  # [B, 4, 28, 28] - VAE latent

        # ============ VAE Decode (Blurry Image) ============
        blurry_image = vae.decode(lowlevel_l1 / 0.18215).sample / 2 + 0.5  # [B, 3, 224, 224]
        blurry_image = blurry_image.clamp(0, 1)

        # 512x512로 resize (Versatile Diffusion 입력)
        blurry_512 = F.interpolate(blurry_image, (512, 512), mode='bilinear', align_corners=False)

        # ============ Versatile Diffusion Reconstruction ============
        recon_tensors = versatile_diffusion_reconstruct(
            versatile_diffusion,
            blurry_512,
            fmri_proj,  # [B, 257, 768]
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            img2img_strength=args.img2img_strength,
            generator=generator,
        )  # [B, 3, 512, 512] Tensor

        # ============ Ground Truth 이미지 (DataLoader에서 이미 로드됨) ============
        gt_tensors = gt_images  # [B, 3, 224, 224] - 이중 로드 제거

        # ============ 배치 단위로 디스크에 저장 ============
        save_recon_batch(recon_tensors, gt_tensors, image_ids, save_dir, subject)

        num_samples += len(image_ids)

        # 메모리 정리
        del recon_tensors, gt_tensors, fmri_proj, lowlevel_l1, blurry_image, blurry_512

    # GPU 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()

    return num_samples


# ============================================================================
# Metric Evaluation
# ============================================================================

def compute_metrics(args, all_recons, all_targets, metrics):
    """
    Reconstruction 품질 평가

    Metrics:
        - PixCorr: Pixel-wise correlation (low-level)
        - SSIM: Structural similarity (low-level)
        - AlexNet_2: AlexNet layer 4 features (low-level)
        - AlexNet_5: AlexNet layer 11 features (mid-level)
        - CLIP: CLIP ViT-L/14 features (high-level semantic)
        - Inception: Inception-v3 features (high-level)
        - EfficientNet: EfficientNet-B1 features (high-level)
        - SwAV: Self-supervised features (high-level)

    2-way identification:
        - 각 reconstruction이 N개 GT 중 자신의 GT와 가장 유사한지 측정
        - Score = (자신보다 유사도가 낮은 GT 개수) / (N-1)
        - 1.0 = 완벽, 0.5 = random chance
    """
    results = {}

    print("\nComputing metrics...")

    # Low-level metrics
    results["PixCorr"] = metrics["pixcorr"](all_recons, all_targets)
    results["SSIM"] = metrics["ssim"](all_recons, all_targets)

    # AlexNet features
    results["AlexNet_2"] = metrics["alexnet2"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["alexnet2"]["model"],
        metrics["alexnet2"]["preprocess"],
        metrics["alexnet2"]["layer"]
    )

    results["AlexNet_5"] = metrics["alexnet5"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["alexnet5"]["model"],
        metrics["alexnet5"]["preprocess"],
        metrics["alexnet5"]["layer"]
    )

    # High-level semantic metrics
    results["CLIP"] = metrics["clip"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["clip"]["model"],
        metrics["clip"]["preprocess"]
    )

    results["Inception"] = metrics["inception"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["inception"]["model"],
        metrics["inception"]["preprocess"]
    )

    results["EfficientNet"] = metrics["efficientnet"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["efficientnet"]["model"],
        metrics["efficientnet"]["preprocess"]
    )

    results["SwAV"] = metrics["swav"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["swav"]["model"],
        metrics["swav"]["preprocess"]
    )

    return results



