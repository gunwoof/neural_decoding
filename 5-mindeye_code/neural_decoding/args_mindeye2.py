import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Configuration")

    ###### Frequently changing settings ######
    parser.add_argument(
        '--mode', type=str, choices=['train', 'inference', 'evaluate'], default='train',
        help="train, inference, evaluate 구분"
    )
    parser.add_argument(
        "--device",type=str,default="cuda:3",
        help='device',
    )
    parser.add_argument(
        "--num_epochs",type=int,default=250, choices=[3,240],
        help="epoch 개수",
    )
    parser.add_argument(
        "--batch_size", type=int, default=160,
        help="Batch size(H100:160, L40:90), if benchmark L40:30",
    )


    ###### scheduler ######
    parser.add_argument(
        "--max_lr",type=float,default=3e-4,
    )
    parser.add_argument(
        "--scheduler_type",type=str,default='cycle',
        choices=['cycle','linear'],
    )
    ####################


    ###### data.py ######
    parser.add_argument(
        "--root_dir", type=str, default="/nas/research/03-Neural_decoding",
        help="Path to the BIDS root."
    )
    parser.add_argument(
        "--fmri_dir", type=str, default="3-bids/derivatives",
        help="Path to the BIDS fmri."
    )
    parser.add_argument(
        "--fmri_detail_dir", type=str, default="beta_hf_dk",
        choices=["b4_roi_zscore","beta_huggingface","beta_hf_dk"],
        help="Path to the BIDS fmri_detail."
    )
    parser.add_argument(
        "--image_dir", type=str, default="4-image/beta",
        help="Path to the BIDS image."
    )
    parser.add_argument(
        "--seed",type=int,default=42,
    )
    parser.add_argument(
        "--is_shuffle",action=argparse.BooleanOptionalAction,default=False,
        help="is shuffle",
    )

    ###### optimizer ######
    parser.add_argument(
        "--optimizer",type=str,default='adamw',
    )
    ####################

    ###### scheduler ######
    parser.add_argument(
        "--max_lr",type=float,default=3e-4,
    )
    parser.add_argument(
        "--scheduler_type",type=str,default='cycle',
        choices=['cycle','linear'],
    )
    parser.add_argument(
        "--num_subjects",type=int,default=3, choices=[1, 3, 7],
        help="subject 개수",
    )
    ####################

    ###### trainer ######
    parser.add_argument(
        "--mixup_pct",type=float,default=0.33,
        help="BiMixCo에서 SoftCLIP로 넘어가는 epoch",
    )
    parser.add_argument(
        "--prior_loss_coefficient",type=float,default=30,
        help="prior loss 계수",
    )
    parser.add_argument(
        "--nce_loss_coefficient",type=float,default=1.,
        help="multiply contrastive loss by this number",
    )
    parser.add_argument(
        "--lowlevel_loss_coefficient",type=float,default=5.,
        help="multiply loss from blurry recons by this number",
    )
    parser.add_argument(
        "--code_dir", type=str, default="5-mindeye_code",
        help="Path to the code."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Path to the output."
    )
    parser.add_argument(
        "--model_name", type=str, default="mindeye2_pretrin",
        help="모델 이름"
    )
    parser.add_argument(
        "--recons_per_sample", type=int, default=1,
        help= "한 frmi로 몇 개 sampling할 지"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=20,
        help= "versatile inference step"
    )
    parser.add_argument(
        "--recon_name", type=str, default="mindeye2_recon",
        help="recon 캐시 이름"
    )
    parser.add_argument(
        "--metrics_name", type=str, default="mindeye2_metric",
        help="metric 결과 이름"
    )
    ####################

    # Jupyter 환경에서는 빈 리스트를 전달해야 실행이 됨
    if any("ipykernel_launcher" in arg for arg in sys.argv):
        args = parser.parse_args([])  
    else:
        args = parser.parse_args()

    return args