"""
ConnecToMind2 Main Entry Point

실행 방법:
    python main.py --device cuda:0 --batch_size 32 --num_epochs 300

    또는 args.py의 default 값 사용:
    python main.py
"""

import os
import numpy as np
import torch
import wandb

from args import parse_args
from data import get_dataloader
from models import get_model
from optimizers import get_optimizer_with_different_lr
from schedulers import get_scheduler
from metrics import get_metric
from trainer import train_evaluate_metric


def main():
    """
    실행 순서:
        1. args 파싱
        2. Seed 설정
        3. WandB 초기화 (optional)
        4. 데이터 로더 생성
        5. 모델 로드
        6. Optimizer, Scheduler 생성
        7. Metric 모델 로드
        8. 학습 시작
    """

    # ============ 1. Args ============
    args = parse_args()
    print(f"\n{'='*60}")
    print("ConnecToMind2 Configuration")
    print(f"{'='*60}")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.max_lr}")
    print(f"  Subjects: {args.subjects}")
    print(f"  Experiment: {args.experiment_name}")
    print(f"{'='*60}\n")

    # ============ 2. Seed 설정 ============
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to {args.seed}")

    # ============ 3. WandB 초기화 ============
    if args.wandb_log:
        wandb.init(
            project="ConnecToMind2",
            name=args.experiment_name,
            config=vars(args)
        )
        print("WandB initialized")

    # ============ 4. 데이터 로더 생성 ============
    print("\nLoading data...")

    # Train loader
    args.mode = 'train'
    train_loader = get_dataloader(args)
    print(f"  Train samples: {len(train_loader.dataset)}")

    # Test loaders (subject별 분리)
    args.mode = 'inference'
    test_loaders = get_dataloader(args)
    total_test = sum(len(loader.dataset) for loader in test_loaders.values())
    print(f"  Test samples: {total_test} (subject별: {', '.join(f'{sub}={len(loader.dataset)}' for sub, loader in test_loaders.items())})")

    # ============ 5. 모델 로드 ============
    print("\nLoading models...")
    models = get_model(args)
    print("  ConnecToMind2 model loaded")
    print("  Stable UnCLIP pipeline loaded")
    print("  VAE loaded")
    print("  ConvNext XL loaded")

    # ============ 6. Optimizer & Scheduler ============
    print("\nSetting up optimizer and scheduler...")
    optimizer = get_optimizer_with_different_lr(args, models["connectomind2"])
    lr_scheduler = get_scheduler(args, optimizer, len(train_loader.dataset))
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Scheduler: {args.scheduler_type}")

    # ============ 7. Metric 모델 로드 ============
    print("\nLoading metric models...")
    metrics = get_metric(args)
    print("  Metrics loaded: PixCorr, SSIM, AlexNet, CLIP, Inception, EfficientNet, SwAV")

    # ============ 8. 학습 시작 ============
    print("\nStarting training...")
    trained_model = train_evaluate_metric(
        args=args,
        train_loader=train_loader,
        test_loaders=test_loaders,  # subject별 분리된 dict
        models=models,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metrics=metrics
    )

    # ============ 9. WandB 종료 ============
    if args.wandb_log:
        wandb.finish()
        print("WandB finished")

    print("\nDone!")


if __name__ == "__main__":
    main()
