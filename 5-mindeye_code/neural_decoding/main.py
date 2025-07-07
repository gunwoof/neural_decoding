import os
import gc
import atexit
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
import wandb

from args import parse_args
from data import get_dataloader, sub1_train_dataset, sub1_train_dataset_hug, sub1_train_dataset_FuncSpatial
from mindeye1 import get_model_highlevel, get_model_lowlevel, get_model_highlevel_FuncSpatial
from optimizers import get_optimizer_highlevel, get_optimizer_lowlevel
from schedulers import get_scheduler
from metrics import get_metric
from trainer import train, inference, evaluate, retrieval_evaluate
from all_trainer import high_train_inference_evaluate, low_train_inference_evaluate
from utils import seed_everything, get_unique_path, save_gt_vs_recon_images

def main():
    # parse_args 정의
    args = parse_args()

    if args.mode == "train":
        #### train ####
        # data loader
        seed_everything(args.seed)
        train_data = get_dataloader(args)

        # model 정의
        models = get_model_highlevel(args) 
        model_bundle = {
            "clip": models["clip"].to(args.device),
            "diffusion_prior": models["diffusion_prior"].to(args.device),
        }

        # optimizer 정의
        optimizer = get_optimizer_highlevel(args, model_bundle["diffusion_prior"])

        # scheduler 정의(train만 함)
        train_dataset = sub1_train_dataset(args)
        num_train = len(train_dataset) 
        lr_scheduler = get_scheduler(args, optimizer, num_train)

        # wandb 적용
        wandb.login() # login
        wandb.init(project="neural_decoding", name=f"run-{wandb.util.generate_id()}") # init
        wandb.config = vars(args) # aparse_args()의 내용 그대로 config로 주기

        # train 시작
        output_model = train(args, train_data, model_bundle, optimizer, lr_scheduler)

        # model 저장
        output_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, args.model_name + ".pt")
        output_path = get_unique_path(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 경로 없으면 생성
        torch.save(output_model.state_dict(), output_path)

        # gpu에서 train 비우기
        del train_data, train_dataset, optimizer, lr_scheduler, output_model, model_bundle
        gc.collect()
        torch.cuda.empty_cache()

        setattr(args, 'mode', 'inference')
    
    #### inference ####
    if args.mode == "inference":
        # data loader
        test_data = get_dataloader(args)

        # model 정의
        models = get_model_highlevel_FuncSpatial(args) 
        model_bundle = {
            "clip": models["clip"].to(args.device),
            "diffusion_prior": models["diffusion_prior"].to(args.device),
            "unet": models["unet"].to(args.device), # inference에서만 사용
            "vae": models["vae"].to(args.device), # inference에서만 사용
            "noise_scheduler": models["noise_scheduler"], # inference에서만 사용
        }

        # model불러오기 위해 path저장
        try:
            _ = output_path.shape
        except:
            output_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, 'recon_metric', 'mindeye1_220_fc(1)_learnable_layer1_highx.pt') # mindeye1.pt이 
        
        all_recons, all_targets, save_recons = inference(args, test_data, model_bundle, output_path)

        # inference 저장
        cache_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, args.recon_name + ".pt")  # 저장 경로 설정
        cache_path = get_unique_path(cache_path)  # 중복 방지용 새 경로 생성
        torch.save({"all_recons": all_recons, "all_targets": all_targets}, cache_path)

        # gpu에서 inference 비우기
        del test_data, model_bundle
        gc.collect()
        torch.cuda.empty_cache()

        setattr(args, 'mode', 'evaluate')

    #### evalutate ####
    if args.mode == "evaluate":
        # metric 정의
        metrics = get_metric(args)
        metric_bundle = {
            "pixcorr": metrics["pixcorr"],
            "ssim": metrics["ssim"],
            "alexnet2": {
                "model": metrics["alexnet2"]["model"].to(args.device),
                "preprocess": metrics["alexnet2"]["preprocess"],
                "layer": metrics["alexnet2"]["layer"],
                "metric_fn": metrics["alexnet2"]["metric_fn"],
            },
            "alexnet5": {
                "model": metrics["alexnet5"]["model"].to(args.device),
                "preprocess": metrics["alexnet5"]["preprocess"],
                "layer": metrics["alexnet5"]["layer"],
                "metric_fn": metrics["alexnet5"]["metric_fn"],
            },
            "clip": {
                "model": metrics["clip"]["model"].to(args.device),
                "preprocess": metrics["clip"]["preprocess"],
                "metric_fn": metrics["clip"]["metric_fn"],
            },
            "inception": {
                "model": metrics["inception"]["model"].to(args.device),
                "preprocess": metrics["inception"]["preprocess"],
                "metric_fn": metrics["inception"]["metric_fn"],
            },
            "efficientnet": {
                "model": metrics["efficientnet"]["model"].to(args.device),
                "preprocess": metrics["efficientnet"]["preprocess"],
                "metric_fn": metrics["efficientnet"]["metric_fn"],
            },
            "swav": {
                "model": metrics["swav"]["model"].to(args.device),
                "preprocess": metrics["swav"]["preprocess"],
                "metric_fn": metrics["swav"]["metric_fn"],
            },
        }

        try: 
            _ = all_recons.shape
        except:
            cache_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, "mindeye1_recon_2.pt")
            cache = torch.load(cache_path, map_location="cpu")
            all_recons = cache["all_recons"]
            all_targets = cache["all_targets"]
            
        metric_results = evaluate(args, all_recons, all_targets, metric_bundle)

        # metric 저장
        txt_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, args.metrics_name + ".txt")
        txt_path = get_unique_path(txt_path)
        with open(txt_path, "w") as f:
            for name, score in metric_results.items():
                f.write(f"{name}: {score:.4f}\n")

        # save_recons 저장
        recons_dir = os.path.join(args.root_dir, args.code_dir, args.output_dir, "recon_highx")
        save_gt_vs_recon_images(save_recons, recons_dir)


def main_high_all():
    args = parse_args()

    # data loader
    seed_everything(args.seed) # 시드 고정
    train_data = get_dataloader(args)
    setattr(args, 'mode', 'inference')
    test_data = get_dataloader(args)

    # model 정의
    models = get_model_highlevel(args) 
    model_bundle = {
        "clip": models["clip"].to(args.device),
        "diffusion_prior": models["diffusion_prior"].to(args.device),
        "unet": models["unet"].to(args.device), # inference에서만 사용
        "vae": models["vae"].to(args.device), # inference에서만 사용
        "noise_scheduler": models["noise_scheduler"], # inference에서만 사용
    }

    # optimizer 정의
    optimizer = get_optimizer_highlevel(args, model_bundle["diffusion_prior"])

    # scheduler 정의(train만 함)
    train_dataset = sub1_train_dataset_hug(args)
    num_train = len(train_dataset) 
    lr_scheduler = get_scheduler(args, optimizer, num_train)

    # metric 정의
    metrics = get_metric(args)
    metric_bundle = {
        "pixcorr": metrics["pixcorr"],
        "ssim": metrics["ssim"],
        "alexnet2": {
            "model": metrics["alexnet2"]["model"].to(args.device),
            "preprocess": metrics["alexnet2"]["preprocess"],
            "layer": metrics["alexnet2"]["layer"],
            "metric_fn": metrics["alexnet2"]["metric_fn"],
        },
        "alexnet5": {
            "model": metrics["alexnet5"]["model"].to(args.device),
            "preprocess": metrics["alexnet5"]["preprocess"],
            "layer": metrics["alexnet5"]["layer"],
            "metric_fn": metrics["alexnet5"]["metric_fn"],
        },
        "clip": {
            "model": metrics["clip"]["model"].to(args.device),
            "preprocess": metrics["clip"]["preprocess"],
            "metric_fn": metrics["clip"]["metric_fn"],
        },
        "inception": {
            "model": metrics["inception"]["model"].to(args.device),
            "preprocess": metrics["inception"]["preprocess"],
            "metric_fn": metrics["inception"]["metric_fn"],
        },
        "efficientnet": {
            "model": metrics["efficientnet"]["model"].to(args.device),
            "preprocess": metrics["efficientnet"]["preprocess"],
            "metric_fn": metrics["efficientnet"]["metric_fn"],
        },
        "swav": {
            "model": metrics["swav"]["model"].to(args.device),
            "preprocess": metrics["swav"]["preprocess"],
            "metric_fn": metrics["swav"]["metric_fn"],
        },
    }

    # wandb 적용
    wandb.login() # login
    wandb.init(project="neural_decoding_highlevel", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

    high_train_inference_evaluate(args, train_data, test_data, model_bundle, optimizer, lr_scheduler, metric_bundle)

def main_low_all():
    args = parse_args()

    # data loader
    seed_everything(args.seed, cudnn_deterministic=False)
    train_data = get_dataloader(args)
    setattr(args, 'mode', 'inference')
    test_data = get_dataloader(args)

    models = get_model_lowlevel(args) 
    model_bundle = {
        "voxel2sd": models["voxel2sd"].to(args.device),
        "cnx": models["cnx"].to(args.device),
        "vae": models["vae"].to(args.device),
        "noise_scheduler": models["noise_scheduler"],
    }

    # optimizer 정의
    optimizer = get_optimizer_lowlevel(args, model_bundle["voxel2sd"])

    # scheduler 정의(train만 함)
    train_dataset = sub1_train_dataset_hug(args)
    num_train = len(train_dataset) 
    lr_scheduler = get_scheduler(args, optimizer, num_train)

    # metric 정의
    metrics = get_metric(args)
    metric_bundle = {
        "pixcorr": metrics["pixcorr"],
        "ssim": metrics["ssim"],
        "alexnet2": {
            "model": metrics["alexnet2"]["model"].to(args.device),
            "preprocess": metrics["alexnet2"]["preprocess"],
            "layer": metrics["alexnet2"]["layer"],
            "metric_fn": metrics["alexnet2"]["metric_fn"],
        },
        "alexnet5": {
            "model": metrics["alexnet5"]["model"].to(args.device),
            "preprocess": metrics["alexnet5"]["preprocess"],
            "layer": metrics["alexnet5"]["layer"],
            "metric_fn": metrics["alexnet5"]["metric_fn"],
        },
        "clip": {
            "model": metrics["clip"]["model"].to(args.device),
            "preprocess": metrics["clip"]["preprocess"],
            "metric_fn": metrics["clip"]["metric_fn"],
        },
        "inception": {
            "model": metrics["inception"]["model"].to(args.device),
            "preprocess": metrics["inception"]["preprocess"],
            "metric_fn": metrics["inception"]["metric_fn"],
        },
        "efficientnet": {
            "model": metrics["efficientnet"]["model"].to(args.device),
            "preprocess": metrics["efficientnet"]["preprocess"],
            "metric_fn": metrics["efficientnet"]["metric_fn"],
        },
        "swav": {
            "model": metrics["swav"]["model"].to(args.device),
            "preprocess": metrics["swav"]["preprocess"],
            "metric_fn": metrics["swav"]["metric_fn"],
        },
    }

    # wandb 적용
    wandb.login() # login
    wandb.init(project="neural_decoding_lowlevel", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

    low_train_inference_evaluate(args, train_data, test_data, model_bundle, optimizer, lr_scheduler, metric_bundle)

def main_high_all_FuncSpatial():
    args = parse_args()

    # data loader
    seed_everything(args.seed) # 시드 고정
    train_data = get_dataloader(args)
    setattr(args, 'mode', 'inference')
    test_data = get_dataloader(args)

    # model 정의
    models = get_model_highlevel_FuncSpatial(args) 
    model_bundle = {
        "clip": models["clip"].to(args.device),
        "diffusion_prior": models["diffusion_prior"].to(args.device),
        "unet": models["unet"].to(args.device), # inference에서만 사용
        "vae": models["vae"].to(args.device), # inference에서만 사용
        "noise_scheduler": models["noise_scheduler"], # inference에서만 사용
    }

    # optimizer 정의
    optimizer = get_optimizer_highlevel(args, model_bundle["diffusion_prior"])

    # scheduler 정의(train만 함)
    train_dataset = sub1_train_dataset_FuncSpatial(args)
    num_train = len(train_dataset) 
    lr_scheduler = get_scheduler(args, optimizer, num_train)

    # metric 정의
    metrics = get_metric(args)
    metric_bundle = {
        "pixcorr": metrics["pixcorr"],
        "ssim": metrics["ssim"],
        "alexnet2": {
            "model": metrics["alexnet2"]["model"].to(args.device),
            "preprocess": metrics["alexnet2"]["preprocess"],
            "layer": metrics["alexnet2"]["layer"],
            "metric_fn": metrics["alexnet2"]["metric_fn"],
        },
        "alexnet5": {
            "model": metrics["alexnet5"]["model"].to(args.device),
            "preprocess": metrics["alexnet5"]["preprocess"],
            "layer": metrics["alexnet5"]["layer"],
            "metric_fn": metrics["alexnet5"]["metric_fn"],
        },
        "clip": {
            "model": metrics["clip"]["model"].to(args.device),
            "preprocess": metrics["clip"]["preprocess"],
            "metric_fn": metrics["clip"]["metric_fn"],
        },
        "inception": {
            "model": metrics["inception"]["model"].to(args.device),
            "preprocess": metrics["inception"]["preprocess"],
            "metric_fn": metrics["inception"]["metric_fn"],
        },
        "efficientnet": {
            "model": metrics["efficientnet"]["model"].to(args.device),
            "preprocess": metrics["efficientnet"]["preprocess"],
            "metric_fn": metrics["efficientnet"]["metric_fn"],
        },
        "swav": {
            "model": metrics["swav"]["model"].to(args.device),
            "preprocess": metrics["swav"]["preprocess"],
            "metric_fn": metrics["swav"]["metric_fn"],
        },
    }

    # wandb 적용
    wandb.login() # login
    wandb.init(project="neural_decoding_highlevel", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

    high_train_inference_evaluate(args, train_data, test_data, model_bundle, optimizer, lr_scheduler, metric_bundle)

def retrieval():
    args = parse_args()
    setattr(args, 'mode', 'inference')
    fwds, bwds = [], []
    
    for i in range(30):
    
        test_data = get_dataloader(args)

        # model 정의
        # models = get_model_highlevel(args)
        models = get_model_highlevel_FuncSpatial(args) 
        model_bundle = {
            "clip": models["clip"].to(args.device),
            "diffusion_prior": models["diffusion_prior"].to(args.device),
        }
        output_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, 'recon_metric', 'mindeye1_220_fc(0.7)_learnable_layer1.pt')

        fwd, bwd = retrieval_evaluate(args, test_data, model_bundle, output_path)

        fwds=np.append(fwds, fwd)
        bwds=np.append(bwds, bwd)

    percent_fwd = np.mean(fwds)
    percent_bwd = np.mean(bwds)

    print(f"fwd percent_correct: {percent_fwd:.4f}")
    print(f"bwd percent_correct: {percent_bwd:.4f}")
    
    result_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, f"mindeye1_retrieval_metrics_{args.experiment_name}.txt")
    result_path = get_unique_path(result_path)
    with open(result_path, "w") as f:
        f.write(f"Forward Retrieval Accuracy: {percent_fwd:.4f}\n")
        f.write(f"Backward Retrieval Accuracy: {percent_bwd:.4f}\n")


if __name__ == "__main__":
    main()
    # main_high_all()
    # main_low_all()
    # main_high_all_FuncSpatial()
    # retrieval()