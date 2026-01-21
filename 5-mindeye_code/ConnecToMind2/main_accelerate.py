"""
ConnecToMind2 Main Entry Point with Accelerate (DeepSpeed ZeRO-2)

실행 방법:
    # Single GPU (테스트용)
    python main_accelerate.py

    # Multi-GPU (DeepSpeed ZeRO-2) - config 파일 사용 (권장)
    accelerate launch --config_file configs/accelerate_config.yaml main_accelerate.py

    # Multi-GPU (DeepSpeed ZeRO-2) - 커맨드라인
    accelerate launch --mixed_precision=bf16 --use_deepspeed --num_processes=6 main_accelerate.py

주의사항:
    - batch_size는 GPU당 배치 크기입니다
    - Total effective batch = batch_size × num_processes
    - Learning rate는 자동으로 scaling되지 않으므로 수동 조정 필요
    - DeepSpeed ZeRO-2: optimizer state + gradient를 GPU 간 분산 (메모리 절감 50-60%)
"""

import os
import math
import numpy as np
import torch
import wandb

from accelerate import Accelerator
from accelerate.utils import set_seed

from args import parse_args
from data import get_dataloader
from models import get_model
from optimizers import get_optimizer_with_different_lr
from schedulers import get_scheduler
from metrics import get_metric
from trainer_accelerate import train_evaluate_metric


def main():
    """
    실행 순서:
        1. Accelerator 초기화 (DDP 설정)
        2. args 파싱
        3. Seed 설정 (재현성)
        4. WandB 초기화 (main process only)
        5. 데이터 로더 생성
        6. 모델 로드
        7. Optimizer, Scheduler 생성
        8. Accelerator prepare (DDP wrapping)
        9. Metric 모델 로드 (main process only)
        10. 학습 시작
    """

    # ============ 1. Accelerator 초기화 ============
    # DeepSpeed ZeRO-2: accelerate_config.yaml에서 설정 로드
    # - gradient_clipping: deepspeed_config.json에서 설정 (1.0)
    # - mixed_precision: accelerate_config.yaml에서 설정 (bf16)
    # - zero_stage: 2 (optimizer state + gradient 분산)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="wandb" if False else None,  # WandB tracking (optional)
    )

    # ============ 2. Args ============
    args = parse_args()

    # Accelerator에서 device 자동 할당 (각 process마다 다른 GPU)
    args.device = accelerator.device

    # Learning rate scaling 비활성화 (기본값 유지)
    # args.max_lr = args.max_lr * math.sqrt(accelerator.num_processes)

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print("ConnecToMind2 Configuration (Accelerate DDP)")
        print(f"{'='*60}")
        print(f"  Mode: {args.mode}")
        print(f"  Distributed: {accelerator.num_processes} processes")
        print(f"  Local Rank: {accelerator.local_process_index}")
        print(f"  Device: {args.device}")
        print(f"  Mixed Precision: {accelerator.mixed_precision}")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Batch Size (per GPU): {args.batch_size}")
        print(f"  Total Batch Size: {args.batch_size * accelerator.num_processes}")
        print(f"  Learning Rate: {args.max_lr}")
        print(f"  Subjects: {args.subjects}")
        print(f"  Experiment: {args.experiment_name}")
        print(f"{'='*60}\n")

    # ============ 3. Seed 설정 ============
    set_seed(args.seed)
    if accelerator.is_main_process:
        print(f"Random seed set to {args.seed}")

    # ============ 4. WandB 초기화 (main process only) ============
    if args.wandb_log and accelerator.is_main_process:
        wandb.init(
            project="ConnecToMind2",
            name=args.experiment_name,
            config=vars(args)
        )
        print("WandB initialized")

    # ============ 5. 데이터 로더 생성 ============
    if accelerator.is_main_process:
        print("\nLoading data...")

    # Train + Validation loader
    args.mode = 'train'
    train_loader, val_loader = get_dataloader(args)

    if accelerator.is_main_process:
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")

    # Test loaders (subject별 분리) - DDP로 평가 병렬화
    args.mode = 'inference'
    # Evaluation용 num_workers 줄임 (NCCL 타임아웃 방지)
    original_num_workers = args.num_workers
    args.num_workers = args.eval_num_workers
    test_loaders_raw = get_dataloader(args)
    args.num_workers = original_num_workers  # 복원

    # Test loaders를 DDP로 준비 (각 GPU가 다른 batch 처리)
    test_loaders = {}
    for sub, loader in test_loaders_raw.items():
        test_loaders[sub] = accelerator.prepare(loader)

    if accelerator.is_main_process:
        total_test = sum(len(loader.dataset) for loader in test_loaders_raw.values())
        print(f"  Test samples: {total_test} (subject별: {', '.join(f'{sub}={len(loader.dataset)}' for sub, loader in test_loaders_raw.items())})")
        print(f"  Evaluation will use DDP with {accelerator.num_processes} GPUs (num_workers={args.eval_num_workers})")

    # ============ 6. 모델 로드 ============
    if accelerator.is_main_process:
        print("\nLoading models...")

    models = get_model(args)

    if accelerator.is_main_process:
        print("  ConnecToMind2 model loaded")
        print("  Versatile Diffusion pipeline loaded")
        print("  VAE loaded")
        print("  ConvNext XL loaded")

    # ============ 7. Optimizer & Scheduler ============
    if accelerator.is_main_process:
        print("\nSetting up optimizer and scheduler...")

    optimizer = get_optimizer_with_different_lr(args, models["connectomind2"])
    lr_scheduler = get_scheduler(
        args, optimizer, len(train_loader.dataset),
        num_processes=accelerator.num_processes  # DDP GPU 수 전달
    )

    if accelerator.is_main_process:
        print(f"  Optimizer: {args.optimizer}")
        print(f"  Scheduler: {args.scheduler_type}")

    # ============ 8. Accelerator Prepare ============
    # DDP로 모델, optimizer, dataloader wrap
    # - model: DistributedDataParallel로 wrap
    # - dataloader: DistributedSampler 자동 적용
    # - optimizer: gradient synchronization 설정
    models["connectomind2"], optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        models["connectomind2"], optimizer, train_loader, val_loader, lr_scheduler
    )

    # VAE, Versatile Diffusion, ConvNext는 frozen이므로 각 GPU에 복사만
    # (학습되지 않으므로 DDP 불필요)
    models["vae"] = models["vae"].to(accelerator.device)
    if models["versatile_diffusion"] is not None:
        models["versatile_diffusion"] = models["versatile_diffusion"].to(accelerator.device)
    # NOTE: ConvNext는 현재 비활성화되어 있음
    # models["cnx"] = models["cnx"].to(accelerator.device)

    # Test loaders는 DDP 필요 없음 (evaluation은 main process만 수행)

    if accelerator.is_main_process:
        print("  Models prepared for distributed training")
        print(f"  Trainable parameters: {sum(p.numel() for p in models['connectomind2'].parameters() if p.requires_grad):,}")

    # ============ 9. Metric 모델 로드 (main process only) ============
    metrics = None
    if accelerator.is_main_process:
        print("\nLoading metric models...")
        metrics = get_metric(args)
        print("  Metrics loaded: PixCorr, SSIM, AlexNet, CLIP, Inception, EfficientNet, SwAV")

    # ============ 10. 학습 시작 ============
    if accelerator.is_main_process:
        print("\nStarting training...")

    trained_model = train_evaluate_metric(
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loaders=test_loaders,  # subject별 분리된 dict
        models=models,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metrics=metrics,
        accelerator=accelerator  # 추가!
    )

    # ============ 11. WandB 종료 ============
    if args.wandb_log and accelerator.is_main_process:
        wandb.finish()
        print("WandB finished")

    if accelerator.is_main_process:
        print("\nTraining Complete!")


if __name__ == "__main__":
    main()
