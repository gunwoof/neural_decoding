import torch
import math


def linear_scheduler(optimizer, total_steps):
    """LinearLR: 학습률을 선형적으로 감소"""
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=total_steps,
        last_epoch=-1
    )


def cycle_scheduler(optimizer, total_steps, max_lr, num_epochs):
    """OneCycleLR: warmup + decay 사이클 학습률"""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1,
        pct_start=2 / num_epochs  # warmup 비율
    )


def cosine_scheduler(optimizer, total_steps, num_warmup_steps=0):
    """CosineAnnealingLR with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_scheduler(args, optimizer, num_train, num_processes=1):
    """
    학습률 스케줄러 선택자

    Args:
        args: arguments containing scheduler settings
        optimizer: optimizer instance
        num_train: number of training samples
        num_processes: DDP에서 사용하는 GPU 수 (default: 1)

    Returns:
        scheduler: learning rate scheduler
    """
    # total_steps 계산 (DDP에서는 effective batch size = batch_size × num_processes)
    effective_batch_size = args.batch_size * num_processes
    total_steps = int(args.num_epochs * math.ceil(num_train / effective_batch_size))

    if args.scheduler_type == 'linear':
        return linear_scheduler(optimizer, total_steps)

    if args.scheduler_type == 'cycle':
        return cycle_scheduler(optimizer, total_steps, args.max_lr, args.num_epochs)

    if args.scheduler_type == 'cosine':
        num_warmup_steps = int(total_steps * 0.1)  # 10% warmup
        return cosine_scheduler(optimizer, total_steps, num_warmup_steps)

    raise ValueError(f"Unknown scheduler type: {args.scheduler_type}")
