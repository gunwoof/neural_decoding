# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neural decoding research project focused on reconstructing visual images from brain fMRI activity. It implements MindEye1, MindEye2, and ConnecToMind2 models that decode brain voxel activations into image reconstructions using CLIP embeddings and diffusion models.

## Project Structure

```
1-raw_data/          # Raw neuroimaging datasets (NSD, BOLD5000, GOD, THINGS)
2-Bids_code/         # BIDS format conversion and GLM beta extraction scripts
3-bids/              # BIDS-formatted neuroimaging data and derivatives
4-image/             # Image datasets (nsd_beta_img/, coco_beta_img/)
5-mindeye_code/      # Main neural decoding models
  ├── fMRI-reconstruction-NSD/  # MindEye1 (NeurIPS 2023)
  ├── MindEyeV2/                # MindEye2 (ICML 2024) - SOTA
  ├── ConnecToMind2/            # Functional connectivity variant
  ├── neural_decoding/          # Custom unified training framework
  └── pretrained_cache/         # Pre-trained generative models
6-preprocess_data/   # Additional preprocessing utilities
```

## Environment Setup

### MindEye2 (recommended)
```bash
cd 5-mindeye_code/MindEyeV2/src
. setup.sh  # Creates 'fmri' virtual environment with Python 3.11
source fmri/bin/activate
```

### Custom neural_decoding module
```bash
cd 5-mindeye_code/neural_decoding
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.1.0, transformers 4.37.2, diffusers 0.23.0, open_clip, dalle2-pytorch

## Common Commands

### MindEye2 Training
```bash
cd 5-mindeye_code/MindEyeV2/src

# Multi-subject pretraining (exclude subject 1)
accelerate launch --mixed_precision=fp16 Train.py \
  --model_name=final_multisubject_subj01 \
  --multi_subject --subj=1 --batch_size=42 \
  --num_sessions=40 --hidden_dim=4096

# Fine-tune on subject 1
accelerate launch --mixed_precision=fp16 Train.py \
  --model_name=final_subj01_pretrained_40sess_24bs \
  --no-multi_subject --subj=1 --batch_size=24 \
  --multisubject_ckpt=../train_logs/final_multisubject_subj01

# Memory-efficient training (weaker GPUs)
accelerate launch --mixed_precision=fp16 Train.py \
  --hidden_dim=1024 --no-blurry_recon --batch_size=24
```

### MindEye1 Training
```bash
cd 5-mindeye_code/fMRI-reconstruction-NSD/src
python Train_MindEye.py --model_name=my_model --subj=1 --wandb_log
```

### Inference & Evaluation
```bash
# Run inference notebooks or convert to .py
python recon_inference.py --model_name=final_subj01_pretrained_40sess_24bs --subj=1
python Reconstruction_Metrics.py --recon_path=reconstructions.pt --all_images_path=all_images.pt
```

### Data Preprocessing (BIDS)
```bash
cd 2-Bids_code
python bids_nsd.py    # Convert NSD to BIDS format
python bids2beta.py   # Extract GLM beta values from BOLD
```

## Key Architecture Files

- **MindEye1**: `5-mindeye_code/fMRI-reconstruction-NSD/src/models.py`
- **MindEye2**: `5-mindeye_code/MindEyeV2/src/models.py` (contains `BrainNetwork` MLP-Mixer)
- **ConnecToMind2**: `5-mindeye_code/ConnecToMind2/models.py` (uses functional connectivity)
- **Training loops**: `5-mindeye_code/neural_decoding/all_trainer.py`, `all_trainer_mindeye2.py`
- **Data loading**: `5-mindeye_code/neural_decoding/data.py` (`TrainDataset`, `TestDataset`)
- **Arguments**: `5-mindeye_code/neural_decoding/args.py`, `args_mindeye2.py`

## Dataset Information

**Primary**: Natural Scenes Dataset (NSD) - 8 subjects, 7T fMRI, 10,000+ COCO images
- Subjects used in code: 1, 2, 5, 7
- Raw data: `1-raw_data/nsd/`
- Processed betas: `3-bids/2-derivatives/1-beta/`
- Images: `4-image/nsd_beta_img/` (256x256 JPG)

**Data Access**: Requires NSD Terms & Conditions agreement and Data Access form

## Key Hyperparameters

MindEye2 defaults:
- `hidden_dim`: 4096 (use 1024 for memory efficiency)
- `n_blocks`: 4
- `batch_size`: 42 (pretraining), 24 (fine-tuning)
- `num_epochs`: 150
- `prior_scale`: 30, `clip_scale`: 1, `blur_scale`: 0.5

## Data Pipeline

1. **Raw fMRI** → **BIDS format** (`2-Bids_code/bids_nsd.py`)
2. **BOLD time-series** → **GLM beta values** (`2-Bids_code/bids2beta.py`)
3. **Beta values + images** → **CLIP embeddings** → **Image reconstruction**

## Experiment Tracking

Weights & Biases integration available via `--wandb_log` flag. Logs stored in `5-mindeye_code/wandb/`.

## Citations

- MindEye2: arXiv:2403.11207 (ICML 2024)
- MindEye1: arXiv:2305.18274 (NeurIPS 2023)
- Natural Scenes Dataset: Nature Neuroscience (2021)
