import argparse
import sys
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="ConnecToMind2 Training Configuration")

    ###### Mode & Device ######
    parser.add_argument(
        '--mode', type=str, choices=['train', 'inference', 'evaluate'], default='train',
        help="train, inference, evaluate 구분"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:6",
        help='only one gpu device'
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed"
    )

    ###### Training Settings ######
    parser.add_argument(
        "--num_epochs", type=int, default=300,
        help="epoch 개수"
    )
    parser.add_argument(
        "--batch_size", type=int, default=40,
        help="Batch size (H100:64, L40:32)"
    )
    parser.add_argument(
        "--inference_batch_size", type=int, default=40,
        help="train batch size와 동일하게 설정"
    )
    parser.add_argument(
        "--metric_batch_size", type=int, default=1000,
        help="Metric 계산용 batch size (GPU VRAM에 따라 조절: 24GB=32, 16GB=16)"
    )

    ###### DataLoader Settings ######
    parser.add_argument(
        "--prefetch_factor", type=int, default=10,
        help="prefetch factor for dataloader"
    )
    parser.add_argument(
        "--num_workers", type=int, default=10,
        help="num_workers for dataloader"
    )
    parser.add_argument(
        "--eval_num_workers", type=int, default=10,
        help="num_workers for evaluation dataloader (smaller to avoid NCCL timeout)"
    )
    parser.add_argument(
        "--val_size", type=int, default=8000,
        help="Validation set size (train에서 분리)"
    )

    ###### Data Paths ######
    parser.add_argument(
        "--root_dir", type=str, default="/workspace/03-Neural_decoding",
        help="Path to the root directory"
    )
    parser.add_argument(
        "--fmri_dir", type=str, default="3-bids/2-derivatives/1-beta",
        help="Path to fMRI data"
    )
    parser.add_argument(
        "--fmri_detail_dir", type=str, default="connectomind2",
        choices=["connectomind2", "beta_mni_2mm","connectomind1"],
        help="fMRI preprocessing type"
    )
    parser.add_argument(
        "--image_dir", type=str, default="4-image/nsd_beta_img",
        help="Path to image data"
    )
    parser.add_argument(
        "--subjects", type=str, nargs='+', default=["sub-01", "sub-02", "sub-05", "sub-07"],
        help="Subject list (e.g., sub-01 sub-02)"
    )
    parser.add_argument(
        "--roi_suffix", type=str, default='schaefer100',
        choices=['dk', 'destrieux', 'schaefer100', 'schaefer200'],
        help="ROI atlas suffix (e.g., dk, destrieux, schaefer)"
    )

    ###### Model Architecture ######
    parser.add_argument(
        "--seq_len", type=int, default=100,
        help="ROI 개수 (sequence length)"
    )
    parser.add_argument(
        "--input_dim", type=int, default=3291,
        help="각 ROI의 voxel 개수 + padding"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=768,
        help="Embedding dimension (768 for num_heads=12)"
    )
    parser.add_argument(
        "--num_qformer_layers", type=int, default=12,
        help="Connectome-q former layer 개수"
    )
    parser.add_argument(
        "--num_query_tokens", type=int, default=100,
        help="Q-Former query token 개수 (seq_len과 동일)"
    )

    ###### Functional Connectivity ######
    parser.add_argument(
        "--is_fc", action=argparse.BooleanOptionalAction, default=True,
        help="Functional connectivity matrix 사용 유무"
    )

    ###### Loss Weights ######
    parser.add_argument(
        "--fir_weight", type=float, default=1.0,
        help="FIR (fMRI-Image Reconstruction) loss weight"
    )
    parser.add_argument(
        "--fic_weight", type=float, default=1.0,
        help="FIC (fMRI-Image Contrastive) loss weight"
    )
    parser.add_argument(
        "--fim_weight", type=float, default=0.1,
        help="FIM (matching) loss weight"
    )
    parser.add_argument(
        "--lowlevel_weight", type=float, default=0.01,
        help="Low-level loss weight (L1 with VAE latent)"
    )

    ###### Optimizer ######
    parser.add_argument(
        "--optimizer", type=str, default='adamw',
        choices=['adamw', 'adam'],
        help="Optimizer type"
    )
    parser.add_argument(
        "--max_lr", type=float, default=1e-4, # 3e-4, 1e-4, 5e-5
        help="Maximum learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2,
        help="Weight decay"
    )

    ###### Scheduler ######
    parser.add_argument(
        "--scheduler_type", type=str, default='cycle',
        choices=['cycle', 'linear', 'cosine'],
        help="Learning rate scheduler type"
    )

    ###### Cache & Pretrained ######
    parser.add_argument(
        "--cache_dir", type=str, default='/workspace/03-Neural_decoding/5-mindeye_code/ConnecToMind2/cache',
        help="Pretrained model cache directory"
    )

    ###### Output & Logging ######
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Output directory"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="connectomind2_exp",
        help="Experiment name"
    )
    parser.add_argument(
        "--model_name", type=str, default="connectomind2",
        help="Model name for saving"
    )
    parser.add_argument(
        "--recon_name", type=str, default="connectomind2_recon",
        help="Reconstruction cache name"
    )
    parser.add_argument(
        "--metrics_name", type=str, default="connectomind2_metric",
        help="Metrics result name"
    )
    parser.add_argument(
        "--wandb_log", action=argparse.BooleanOptionalAction, default=True,
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--log_grad_every", type=int, default=100,
        help="Log gradient stats every N steps (0 to disable)"
    )

    ###### Inference Settings (Versatile Diffusion) ######
    parser.add_argument(
        "--num_inference_steps", type=int, default=20,
        help="Versatile Diffusion inference steps"
    )
    parser.add_argument(
        "--recons_per_sample", type=int, default=1,
        help="Number of reconstructions per fMRI sample"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--img2img_strength", type=float, default=0.85,
        help="img2img strength (1=no img2img, 0=only lowlevel)"
    )

    # Jupyter 환경에서는 빈 리스트를 전달
    if any("ipykernel_launcher" in arg for arg in sys.argv):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args
