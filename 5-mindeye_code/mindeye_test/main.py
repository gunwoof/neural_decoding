import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../5-mindeye_code/neural_decoding
GEN_DIR  = os.path.abspath(os.path.join(THIS_DIR, "..", "pretrained_cache", "generative_models"))
if GEN_DIR not in sys.path:
    sys.path.insert(0, GEN_DIR)  # 'sgm'을 최상위처럼 인식시키기
import gc
import atexit
import numpy as np

# from accelerate import Accelerator
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
import wandb

# mindeye1 & connectomind
from args import parse_args
from data import get_dataloader, sub1_train_dataset, sub1_train_dataset_hug, sub1_test_dataset_hug, sub1_train_dataset_FuncSpatial
from mindeye1 import get_model_highlevel, get_model_lowlevel, get_model_highlevel_FuncSpatial
from optimizers import get_optimizer_highlevel, get_optimizer_lowlevel
from schedulers import get_scheduler
from metrics import get_metric
from trainer import train, inference, evaluate, retrieval_evaluate
from all_trainer import high_train_inference_evaluate, low_train_inference_evaluate
from utils import seed_everything, get_unique_path, save_gt_vs_recon_images

# mindeye2 (main_high_all()에서는 사용하지 않음 - 주석 처리)
# from args_mindeye2 import parse_args2
# from data import get_dataloader_hug2, train_dataset_hug2
# from mindeye2 import get_pretrain_model, get_model
# from optimizers import get_optimizer_mindeye2
# from all_trainer_mindeye2 import pre_train, fine_tunning_train, inference_evaluate, pre_train_continous

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

    # data loader - train
    seed_everything(args.seed) # 시드 고정
    train_data = get_dataloader(args)

    # data loader - test (두 가지 버전)
    setattr(args, 'mode', 'inference')

    # Test 1: NPZ beta_mni (3000개, 3번 반복)
    from data import hug_TestDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms

    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    image_dir = args.image_dir
    code_dir = args.code_dir
    output_dir = args.output_dir
    transform = transforms.ToTensor()
    use_low_image = args.use_low_image

    # Test 1: NPY test each (connectomind2 - 3000개, 3번 반복)
    import numpy as np
    image_path = f"{root_dir}/{image_dir}"
    low_image_path = f"{root_dir}/{code_dir}/{output_dir}/mindeye1_metric_ourdata/low_recons_test"

    # connectomind2의 test each (3000개, 3번 반복)
    test_each_beta_path = f"{root_dir}/{fmri_dir}/connectomind2/sub-01/sub-01_beta-test_nsdgeneral_3000.npy"
    test_each_stimuli_path = f"{root_dir}/{fmri_dir}/connectomind2/sub-01/sub-01_stimuli-test_3000.npy"
    test_dataset_each = hug_TestDataset(test_each_beta_path, test_each_stimuli_path, image_path, low_image_path, transform, use_low_image)
    test_data_each = DataLoader(test_dataset_each, batch_size=args.inference_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, prefetch_factor=args.prefetch_factor)

    # Test 2: NPY test avg (connectomind2 - 1000개, 평균)
    test_avg_beta_path = f"{root_dir}/{fmri_dir}/connectomind2/sub-01/sub-01_beta-test_nsdgeneral.npy"
    test_avg_stimuli_path = f"{root_dir}/{fmri_dir}/connectomind2/sub-01/sub-01_stimuli-test.npy"
    test_dataset_avg = hug_TestDataset(test_avg_beta_path, test_avg_stimuli_path, image_path, low_image_path, transform, use_low_image)
    test_data_avg = DataLoader(test_dataset_avg, batch_size=args.inference_batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True, prefetch_factor=args.prefetch_factor)

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
    wandb.init(project="mindeye_first_dual_test", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

    # 두 가지 test set으로 평가 (리스트로 전달)
    test_data_list = [
        ("test_each_3000", test_data_each),  # (이름, dataloader) - 3번 반복
        ("test_avg_1000", test_data_avg)     # 평균
    ]
    high_train_inference_evaluate(args, train_data, test_data_list, model_bundle, optimizer, lr_scheduler, metric_bundle)

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
    wandb.init(project="neural_decoding_highlevel_test", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

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

def main_mindeye2_pretrain():

    args = parse_args2()
    device = args.device

    # data loader
    subj_names = [ 'sub-01', 'sub-02', 'sub-05', 'sub-07']
    seed_everything(args.seed) # 시드 고정
    train_data = get_dataloader_hug2(args, subj_names)

    # model 정의
    models = get_pretrain_model(args) 
    model_bundle = {
        "clip": models["clip"].to(device),
        "mindeye2": models["mindeye2"].to(device),
        "vae": models["vae"].to(device), 
        "cnx": models["cnx"].to(device), 
        "l1":  models["l1"].to(device)
    }


    # optimizer 정의
    optimizer = get_optimizer_mindeye2(args, model_bundle["mindeye2"])

    # scheduler 정의(train만 함)
    train_dataset = train_dataset_hug2(args, subj_names)
    num_train = sum(len(dataset) for dataset in train_dataset.values()) # subject별 dict
    lr_scheduler = get_scheduler(args, optimizer, num_train)

    # wandb 적용
    wandb.login() # login
    wandb.init(project="mindeye2_pretrain", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

    # train 시작
    pre_train(args, subj_names, train_data, model_bundle, optimizer, lr_scheduler)

def main_mindeye2_pretrain_continous():

    args = parse_args2()
    device = args.device

    # data loader
    subj_names = [ 'sub-01', 'sub-02', 'sub-05', 'sub-07']
    seed_everything(args.seed) # 시드 고정
    train_data = get_dataloader_hug2(args, subj_names)

    # model 정의
    models = get_pretrain_model(args) 
    model_bundle = {
        "clip": models["clip"].to(device),
        "mindeye2": models["mindeye2"].to(device),
        "vae": models["vae"].to(device), 
        "cnx": models["cnx"].to(device), 
        "l1":  models["l1"].to(device)
    }


    # optimizer 정의
    optimizer = get_optimizer_mindeye2(args, model_bundle["mindeye2"])

    # scheduler 정의(train만 함)
    train_dataset = train_dataset_hug2(args, subj_names)
    num_train = sum(len(dataset) for dataset in train_dataset.values()) # subject별 dict
    lr_scheduler = get_scheduler(args, optimizer, num_train)

    # wandb 적용
    wandb.login() # login
    wandb.init(project="mindeye2_pretrain_continous", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

    # train 시작
    pre_train_continous(args, subj_names, train_data, model_bundle, optimizer, lr_scheduler)


def main_mindeye2_finetunning_train():

    args = parse_args2()
    device = args.device

    # data loader
    subj_names = ['sub-01']
    seed_everything(args.seed) # 시드 고정
    train_data = get_dataloader_hug2(args, subj_names)

    # model 정의
    models = get_model(args) 
    model_bundle = {
        "clip": models["clip"].to(device),
        "mindeye2": models["mindeye2"].to(device),
        "vae": models["vae"].to(device), 
        "cnx": models["cnx"].to(device), 
        "l1":  models["l1"].to(device),
    }

    # optimizer 정의
    optimizer = get_optimizer_mindeye2(args, model_bundle["mindeye2"])

    # scheduler 정의(train만 함)
    train_dataset = train_dataset_hug2(args, subj_names)
    num_train = sum(len(dataset) for dataset in train_dataset.values()) # subject별 dict
    lr_scheduler = get_scheduler(args, optimizer, num_train)

    # wandb 적용
    wandb.login() # login
    wandb.init(project="mindeye2_finetunning_train", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

    # train 시작
    fine_tunning_train(args, subj_names, train_data, model_bundle, optimizer, lr_scheduler)

def main_mindeye2_inference_evaluate():

    args = parse_args2()
    device = args.device

    # data loader
    subj_names = ['sub-01']
    seed_everything(args.seed) # 시드 고정
    setattr(args, 'mode', 'inference')
    test_data = get_dataloader_hug2(args, subj_names)

    # model 정의
    models = get_model(args) 
    model_bundle = {
        "mindeye2": models["mindeye2"].to(device),
        "vae": models["vae"].to(device), 
        "noise_scheduler": models["noise_scheduler"], 
        "clip_linear": models["clip_linear"].to(device), # inference에서만 사용
        "clip_text_model": models["clip_text_model"].to(device), # inference에서만 사용
        "token_to_text": models["token_to_text"], # inference에서만 사용
        "base_text_embedder1": models["base_text_embedder1"].to(device), # inference에서만 사용
        "base_text_embedder2": models["base_text_embedder2"].to(device), # inference에서만 사용
        "sdxl": models["sdxl"].to(device), # inference에서만 사용
        "sdxl_unclip": models["sdxl_unclip"].to(device) # inference에서만 사용
    }

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

    # checkpoint path저장
    ckpt_dir = os.path.join(args.root_dir, args.code_dir, args.output_dir, "mindeye2_metric")

    # wandb 적용
    wandb.login() # login
    wandb.init(project="mindeye2_inference_evaluate", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

    # infernce + evaluate 시작
    inference_evaluate(args, subj_names, test_data, model_bundle, metric_bundle, ckpt_dir)


if __name__ == "__main__":
    # main()
    main_high_all()
    # main_low_all()
    # main_high_all_FuncSpatial()
    # retrieval()
    # main_mindeye2_pretrain()
    # main_mindeye2_finetunning_train()
    # main_mindeye2_inference_evaluate()
    # main_mindeye2_pretrain_continous()

