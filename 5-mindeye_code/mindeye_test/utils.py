import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import wandb
import gc

# image augmentation
import kornia
from kornia.augmentation.container import AugmentationSequential

# mixco
import torch.nn.functional as F

# from pretrained_cache.generative_models.sgm.util import append_dims  # MindEye2에서만 사용 - main_high_all()에서는 불필요

# seed
def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

# image augmentation
def img_augment_high(image: torch.Tensor, version=2):
    if version == 1:
        img_augment_pipeline = AugmentationSequential(
            kornia.augmentation.RandomResizedCrop((224,224), (0.6,1), p=0.3),
            kornia.augmentation.Resize((224, 224)),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
            kornia.augmentation.RandomGrayscale(p=0.3),
            data_keys=["input"],
        )
    if version == 2:
        img_augment_pipeline = AugmentationSequential(
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
            same_on_batch=False,
            data_keys=["input"],
        )
    augmented = img_augment_pipeline(image)

    return augmented

# image augmentation
def img_augment_low(image: torch.Tensor, version=2):
    if version == 1:
        img_augment_pipeline = AugmentationSequential(
            # kornia.augmentation.RandomCrop((480, 480), p=0.3),
            # kornia.augmentation.Resize((512, 512)),
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.2),
            kornia.augmentation.RandomSolarize(p=0.2),
            kornia.augmentation.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0), p=0.1),
            kornia.augmentation.RandomResizedCrop((512, 512), scale=(0.5, 1.0)),
            data_keys=["input"],
        )
    if version == 2:
        img_augment_pipeline = AugmentationSequential(
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.1),
            kornia.augmentation.RandomSolarize(p=0.1),
            kornia.augmentation.RandomResizedCrop((224,224), scale=(.9,.9), ratio=(1,1), p=1.0),
            data_keys=["input"],
        )
    augmented = img_augment_pipeline(image)

    return augmented

# mix up(배치에서 무작위로 고른 1개 샘플과만 섞음)
def mixup(fmri_vol, beta=0.15, s_thresh=0.5): # ex) fmri_vol.shape: [B, num_voxels]
    perm = torch.randperm(fmri_vol.shape[0])
    fmri_vol_shuffle = fmri_vol[perm].to(fmri_vol.device,dtype=fmri_vol.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([fmri_vol.shape[0]]).to(fmri_vol.device,dtype=fmri_vol.dtype)
    select = (torch.rand(fmri_vol.shape[0]) <= s_thresh).to(fmri_vol.device)
    betas_shape = [-1] + [1]*(len(fmri_vol.shape)-1)
    fmri_vol[select] = fmri_vol[select] * betas[select].reshape(*betas_shape) + \
        fmri_vol_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return fmri_vol, perm, betas, select # perm, betas, select는 loss계산할 때 사용

# mixco_nce loss
def mixco_nce_loss(prediction, target, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (prediction @ target.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(prediction.shape[0]).to(prediction.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss

# cosine annealing scheduling
def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

# soft_clip loss
def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

# top-k 중에 정답이 포함되어 있는지 판단
def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

# 상관계수 행렬처럼 target과 prediction간의 cosine_similarity 행렬 -> [target 개수, prediction 개수] 크기의 similarity matrix
def batchwise_cosine_similarity(Z,B):
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

# 파일 저장할때 뒤에 자동증가 숫자 붙이기
def get_unique_path(base_path):
    """
    중복되지 않는 파일 경로를 반환.
    예: diffusion_prior.pth → diffusion_prior_1.pth → diffusion_prior_2.pth ...
    """
    if not os.path.exists(base_path):
        return base_path

    base, ext = os.path.splitext(base_path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"

# 학습할때 nan체크 + 넘김
def check_nan_and_log(
    global_step,
    fmri_vol=None,
    clip_voxels=None,
    voxel_backbone=None,
    voxel_retrieval=None,
    voxel_lowlevels=None,
    loss=None,
    wandb=None
):
    nan_flag = False

    if fmri_vol is not None and torch.isnan(fmri_vol).any():
        print(f"[NaN] Detected in `fmri_vol` at step {global_step}")
        if wandb: wandb.log({"debug/nan_fmri_vol": global_step})
        nan_flag = True

    if clip_voxels is not None and torch.isnan(clip_voxels).any():
        print(f"[NaN] Detected in `clip_voxels` at step {global_step}")
        if wandb: wandb.log({"debug/nan_clip_voxels": global_step})
        nan_flag = True

    if voxel_backbone is not None and torch.isnan(voxel_backbone).any():
        print(f"[NaN] Detected in `voxel_backbone` at step {global_step}")
        if wandb: wandb.log({"debug/nan_voxel_backbone": global_step})
        nan_flag = True

    if voxel_retrieval is not None and torch.isnan(voxel_retrieval).any():
        print(f"[NaN] Detected in `voxel_retrieval` at step {global_step}")
        if wandb: wandb.log({"debug/nan_voxel_retrieval": global_step})
        nan_flag = True

    # tuple이라 다르게 처리
    if voxel_lowlevels is not None and any(torch.isnan(t).any() for t in voxel_lowlevels):
        print(f"[NaN] Detected in `voxel_lowlevels` at step {global_step}")
        if wandb: wandb.log({"debug/nan_voxel_lowlevels": global_step})
        nan_flag = True

    if loss is not None and torch.isnan(loss):
        print(f"[NaN] Detected in `loss` at step {global_step}")
        if wandb: wandb.log({"debug/nan_loss": global_step})
        nan_flag = True

    if nan_flag:
        print(f"[Warning] Skipping batch due to NaN at step {global_step}")

    return nan_flag


def log_gradient_norms(model, global_step=None, verbose=True):
    """
    모델의 각 파라미터 gradient의 L2 norm을 출력하고 전체 norm을 반환합니다.
    
    Args:
        model (torch.nn.Module): gradient를 확인할 모델 (예: diffusion_prior)
        global_step (int, optional): 현재 학습 step (출력용)
        verbose (bool): True면 출력, False면 출력하지 않음

    Returns:
        total_grad_norm (float): 전체 gradient의 L2 norm
    """
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            if verbose:
                print(f"[!] {name} grad is None!")
        elif torch.all(param.grad == 0):
            if verbose:
                print(f"[!] {name} grad is all zeros.")
        else:
            grad_norm = param.grad.data.norm(2).item()
            if verbose:
                print(f"[Grad] {name}: {grad_norm:.6f}")
            total_grad_norm += grad_norm ** 2

    total_grad_norm = total_grad_norm ** 0.5
    if verbose:
        step_msg = f" Step {global_step}" if global_step is not None else ""
        print(f"[Total Grad Norm]{step_msg}: {total_grad_norm:.6f}")
    return total_grad_norm

# reconstruction
@torch.no_grad()
def reconstruction(
    brain_clip_embeddings, proj_embeddings, image,
    clip_extractor, unet=None, vae=None, noise_scheduler=None,
    seed = 42,
    device = "cuda",
    num_inference_steps = 50,
    recons_per_sample = 1, # mindeye에서는 16개
    inference_batch_size=1, # batch 중에서 몇 개만 저장할지 -> batch와 같이 줄 것
    img_lowlevel = None, # low level image
    guidance_scale = 3.5, # 기본 7.5
    img2img_strength = .85,
    plotting=True,
):
    #### setting ####
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if unet:
        # CFG 사용
        do_classifier_free_guidance = guidance_scale > 1.0 
        # resolution 비율: down sampling, up sampling 비율율
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) 
        # 생성할 iamge resolution
        height = unet.config.sample_size * vae_scale_factor 
        width = unet.config.sample_size * vae_scale_factor

    # brain_clip_embeddings [b, 257, 768] → [b*r, 257, 768]
    brain_clip_embeddings = brain_clip_embeddings.repeat_interleave(recons_per_sample, dim=0)  # [b*r, 257, 768]
    total_samples = inference_batch_size * recons_per_sample

    #### versatile diffusion ####
    # cls token norm을 사용하여 모든 patch normalization
    for samp in range(len(brain_clip_embeddings)):
        brain_clip_embeddings[samp] = brain_clip_embeddings[samp]/(brain_clip_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)  
    
    # versatile difussion에 사용할 embedding 정의
    input_embedding = brain_clip_embeddings
    prompt_embeds = torch.zeros(len(input_embedding),77,768) # 사용하지 않을 거지만 difussion.unet.DualTransformer2DModel을 사용하기 위해 필요 -> 모두 0으로 채운 것을 넣기
    
    # CFG 준비
    if do_classifier_free_guidance:
        input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
        prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype) # [2 * b, 77, 768]
    
    # dual_prompt_embeddings [2*b*r, 257+77, 768]
    input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1) 
    
    # CFG로 인해 batch size가 2배 늘어난 것을 2배 나눠야 한다 - low level 있으면 사용
    shape = (inference_batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor) # [b, 257+77, 768]
    
    # timesteps 정의
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    
    # using low level image vs using pure noise
    if img_lowlevel is not None: # low level image에서 시작
        # low level image를 주기 때문에 denoise step을 줄임
        init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = noise_scheduler.timesteps[t_start:]

        # denoise 시작 step 구하기 ex) step start는 scalar라서 inference_batch_size만큼 만듬
        latent_timestep = timesteps[:1].repeat(inference_batch_size) 
        
        # low level image를 vae 통과
        img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
        init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
        init_latents = vae.config.scaling_factor * init_latents
        init_latents = init_latents.repeat_interleave(recons_per_sample, dim=0)

        # low level image에 noise 씌움 = versatile 준비 완료
        noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                            generator=generator, dtype=input_embedding.dtype)
        init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
        latents = init_latents
    else: # pure noise에서 시작 = versatile 준비 완료
        timesteps = noise_scheduler.timesteps
        latents = torch.randn([total_samples, 4, 64, 64], device=device,
                                generator=generator, dtype=input_embedding.dtype)
        latents = latents * noise_scheduler.init_noise_sigma

    # inference - (Denoising loop) 
    for i, t in enumerate(timesteps):
        # cfg를 사용하기 위해 gaussian noise를 condition용 + uncondition용을 만든다
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        
        # noise 예측
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_context = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_context - noise_pred_uncond)

        # compute denoise(x_t) -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    # vae decoder를 통해 image로 변환 # [b*r, 3, H, W]
    recons = decode_latents(latents,vae).detach().cpu()

    brain_recons = recons.view(inference_batch_size, recons_per_sample, 3, height, width) # [b, r, 3, height, width] -> [b, r, 3, height, width]

        
    #### pick best reconstruction out of several ####
    best_picks = np.zeros(inference_batch_size).astype(np.int16)  # best reconstruction 인덱스를 담은 vector
    v2c_reference_out = F.normalize(proj_embeddings.view(len(proj_embeddings), -1), dim=-1)  # [b, 768]
    for sample_idx in range(inference_batch_size):  # inference_batch_size
        sims = []
        reference = v2c_reference_out[sample_idx:sample_idx+1]  # [1, clip_dim]

        for recon_idx in range(recons_per_sample):
            currecon = brain_recons[sample_idx, recon_idx].unsqueeze(0).float()  # [1, 3, H, W]
            currecon = clip_extractor.embed_image(currecon).to(proj_embeddings.device).to(proj_embeddings.dtype)  # [1, clip_dim]
            currecon = F.normalize(currecon.view(len(currecon), -1), dim=-1)  # normalize

            cursim = batchwise_cosine_similarity(reference, currecon)  # (1, 1) similarity
            sims.append(cursim.item())  # scalar 값만 append

        best_picks[sample_idx] = int(np.nanargmax(sims))  # sample_idx 위치에 best recon index 저장

    #### plot ####
    img2img_samples = 0 if img_lowlevel is None else 1
    num_xaxis_subplots = 1 + img2img_samples + recons_per_sample
    best_img = torch.zeros((inference_batch_size, 3, height, width), dtype=brain_recons.dtype) # 초기화

    if plotting:
        fig, ax = plt.subplots(inference_batch_size, num_xaxis_subplots, 
                            figsize=(num_xaxis_subplots*5, 6*inference_batch_size),
                            facecolor=(1, 1, 1))
        for recon_idx in range(inference_batch_size):
            # ax가 1D array일 수도, 2D array일 수도 있음
            axis_row = ax[recon_idx] if inference_batch_size > 1 else ax

            axis_row[0].set_title(f"Original Image")
            axis_row[0].imshow(torch_to_Image(image[recon_idx]))

            if img2img_samples == 1:
                axis_row[1].set_title(f"Img2img ({img2img_strength})")
                axis_row[1].imshow(torch_to_Image(img_lowlevel[recon_idx].clamp(0, 1)))

            for ii, subplot_idx in enumerate(range(num_xaxis_subplots - recons_per_sample, num_xaxis_subplots)):
                recon = brain_recons[recon_idx][ii]
                if ii == best_picks[recon_idx]:
                    axis_row[subplot_idx].set_title(f"Reconstruction", fontweight='bold')
                    best_img[recon_idx] = recon
                else:
                    axis_row[subplot_idx].set_title(f"Recon {ii+1} from brain")
                axis_row[subplot_idx].imshow(torch_to_Image(recon))

            for subplot in axis_row:
                subplot.axis('off')
    else:
        fig = None
        best_img = brain_recons[range(inference_batch_size), best_picks]

    # gpu memory관리
    del latents, input_embedding, prompt_embeds, noise_pred, currecon
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
    return (fig,          # 전체 subplot figure
            brain_recons, # 모든 복원 결과 tensor ex) shape [b, recons_per_sample, 3, height, width]
            best_picks,   # best reconstruction 인덱스 ex) [0번 batch의 가장 좋은 index, 1번 batch의 가장 좋은 index, ...]
            best_img)    # best reconstruction 이미지 ex) [0번 batch의 가장 좋은 image, 1번 batch의 가장 좋은 image, ...]

def decode_latents(latents,vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def plot_best_vs_gt_images(best_imgs, gt_imgs, index, save_dir="outputs", max_imgs=10):
    """
    매 index마다 원본 이미지(gt)와 복원 이미지(best)를 나란히 시각화하여 저장
    - best_imgs: list of [3, H, W] tensors
    - gt_imgs: list of [3, H, W] tensors (ground truth)
    - index: 현재 batch 인덱스 (파일 이름에 사용됨)
    - save_dir: 저장할 디렉토리
    - max_imgs: 최대 시각화할 이미지 수
    """
    os.makedirs(save_dir, exist_ok=True)

    best_imgs = best_imgs[:max_imgs]
    gt_imgs = gt_imgs[:max_imgs]
    n = len(best_imgs)

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))

    for i in range(n):
        axes[i, 0].imshow(torch_to_Image(gt_imgs[i]))
        axes[i, 0].set_title(f"GT {i}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(torch_to_Image(best_imgs[i]))
        axes[i, 1].set_title(f"Recon {i}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"best_vs_gt_batch_{index:03d}.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def save_gt_vs_recon_images(save_recons, save_dir):
    """
    GT와 Recon 이미지를 나란히 저장하는 함수

    Args:
        save_recons (dict): {img_id: (recon_img, gt_img)} 형태의 dict
        save_dir (str): 저장할 디렉토리
    """
    def tensor_to_image(tensor):
        tensor = tensor.detach().cpu().clamp(0, 1)
        return tensor.permute(1, 2, 0).numpy()

    os.makedirs(save_dir, exist_ok=True)

    for img_id, (recon_img, gt_img) in save_recons.items():
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(tensor_to_image(gt_img))
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(tensor_to_image(recon_img))
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")

        plt.tight_layout()

        # 저장
        save_path = os.path.join(save_dir, f"{img_id}")
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    print(f"[완료] {len(save_recons)}개 이미지 저장 완료: {save_dir}")
    
def save_gt_vs_recon_images_extended( all_targets, all_recons, all_blurryrecons, all_enhanced_recons, all_final_recons, all_image_ids,save_dir, layout="horizontal"):
    """
    GT와 4종류의 재구성 이미지를 한 장에 모아 저장하는 함수 (Tensor 기반)

    Args:
        all_targets (torch.Tensor): [N, 3, H, W]
        all_recons (torch.Tensor): [N, 3, H, W]
        all_blurryrecons (torch.Tensor): [N, 3, H, W]
        all_enhanced_recons (torch.Tensor): [N, 3, H, W]
        all_final_recons (torch.Tensor): [N, 3, H, W]
        all_image_ids (torch.Tensor): [N] 또는 list[int/str]
        save_dir (str): 저장할 디렉토리
        layout (str): 'horizontal' 또는 'vertical'
    """

    os.makedirs(save_dir, exist_ok=True)

    def tensor_to_image(t):
        t = t.detach().cpu().clamp(0, 1)
        return t.permute(1, 2, 0).numpy()

    titles = ["Ground Truth", "Recon", "Blurry", "Enhanced", "Final"]

    num_images = all_targets.shape[0]
    for idx in range(num_images):
        imgs = [
            tensor_to_image(all_targets[idx]),
            tensor_to_image(all_recons[idx]),
            tensor_to_image(all_blurryrecons[idx]),
            tensor_to_image(all_enhanced_recons[idx]),
            tensor_to_image(all_final_recons[idx]),
        ]

        if layout == "horizontal":
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        else:
            fig, axes = plt.subplots(5, 1, figsize=(3, 15))

        for i, ax in enumerate(axes):
            ax.imshow(imgs[i])
            ax.set_title(titles[i], fontsize=8)
            ax.axis("off")

        plt.tight_layout()

        # 🔹 image_id가 torch.Tensor이면 int로 변환
        img_id = all_image_ids[idx].item() if torch.is_tensor(all_image_ids[idx]) else all_image_ids[idx]
        save_path = os.path.join(save_dir, f"{img_id}_comparison.png")

        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    print(f"[완료] {num_images}개 이미지 저장 완료: {save_dir}")

def soft_cont_loss(student_preds, teacher_preds, teacher_aug_preds, temp=0.125, distributed=False):
    
    if not distributed:
        teacher_teacher_aug = (teacher_preds @ teacher_aug_preds.T)/temp
        teacher_teacher_aug_t = (teacher_aug_preds @ teacher_preds.T)/temp
        student_teacher_aug = (student_preds @ teacher_aug_preds.T)/temp
        student_teacher_aug_t = (teacher_aug_preds @ student_preds.T)/temp
    else:
        all_student_preds, all_teacher_preds = gather_features(student_preds, teacher_preds)
        all_teacher_aug_preds = gather_features(teacher_aug_preds, None)

        teacher_teacher_aug = (teacher_preds @ all_teacher_aug_preds.T)/temp
        teacher_teacher_aug_t = (teacher_aug_preds @ all_teacher_preds.T)/temp
        student_teacher_aug = (student_preds @ all_teacher_aug_preds.T)/temp
        student_teacher_aug_t = (teacher_aug_preds @ all_student_preds.T)/temp
    
    loss1 = -(student_teacher_aug.log_softmax(-1) * teacher_teacher_aug.softmax(-1)).sum(-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(-1) * teacher_teacher_aug_t.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss


def unclip_recon(x, diffusion_engine, vector_suffix, num_samples=1, offset_noise_level=0.04, device="cuda"):
    from pretrained_cache.generative_models.sgm.util import append_dims  # MindEye2 전용 - 필요할 때만 import

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), diffusion_engine.ema_scope():
        batch_size = x.shape[0]
        z = torch.randn(batch_size,4,96,96).to(device) # starting noise, can change to VAE outputs of initial image for img2img

        # cfg 준비하기 위해 c와 uc 준비
        # 또한 stable diffusion에서 조건은 crossattn와 vector(전역조건) 2개인데 shape가 다름
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}
        tokens = torch.randn_like(x)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        # noise 생성
        noise = torch.randn_like(z) # random값으로 초기화
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps) # 각 step의 추가될 noise 만들어 놓음
        sigma = sigmas[0].to(device)

        # 일반적인 gaussion noise에 추가 noise를 더하는 잡기술
        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )

        # XT(완전한 gaussian noise) 생성: z + sigma*epsilon
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        # denosing 샘플링
        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc) # XT -> X0
        samples_x = diffusion_engine.decode_first_stage(samples_z) # VAE decoder 통과
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)
        return samples
    
def sdxl_recon(inference_batch_size, image, prompt, base_engine, base_text_embedder1, base_text_embedder2, vector_suffix, crossattn_uc, vector_uc, num_samples=1, img2img_timepoint=13, device="cuda"):
    """
    SDXL base engine으로 coarse reconstruction을 refinement/upscale 하는 함수.

    Args:
        image (Tensor): coarse reconstruction, shape (B,3,H,W), 값 범위 [0,1]
        prompt (list[str]): caption prompt 리스트, 길이 B
        base_engine: SDXL base DiffusionEngine
        base_text_embedder1, base_text_embedder2: 텍스트 인코더
        vector_suffix: conditioner에서 나온 전역 vector, shape (B,1024)
        crossattn_uc, vector_uc: unconditional embedding (CFG용)
        num_samples (int): 샘플링할 이미지 수
        img2img_timepoint (int): 얼마나 noisy하게 다시 시작할지 (클수록 coarse하게 재샘플링)
        device (str): 실행 디바이스

    Returns:
        samples (Tensor): refinement된 이미지, shape (B,3,H,W)
    """
    from pretrained_cache.generative_models.sgm.util import append_dims  # MindEye2 전용 - 필요할 때만 import

    with torch.no_grad(), base_engine.ema_scope():
        
        # 1. VAE encode
        z = base_engine.encode_first_stage(image.to(device) * 2 - 1)  # (B,4,H',W')
        z = z.repeat(num_samples, 1, 1, 1)                           # (num_samples*B,4,H',W')

        # 2. 텍스트 condition 준비
        openai_clip_text = base_text_embedder1(prompt)                       # (B, seq_len1, dim)
        clip_text_tokenized, clip_text_emb = base_text_embedder2(prompt)     # (B, seq_len2, dim), (B, dim)
        clip_text_emb = torch.hstack((clip_text_emb, vector_suffix))         # (B, dim+1024)

        # 여기서는 cat 대신 stack/concat 주의: crossattn은 seq 방향 concat, vector는 단순 확장
        clip_text_tokenized = torch.cat((openai_clip_text, clip_text_tokenized), dim=-1)  # (B, seq_len1+seq_len2, dim)

        c = {
            "crossattn": clip_text_tokenized.repeat(num_samples, 1, 1),  # (num_samples*B, total_seq, dim)
            "vector": clip_text_emb.repeat(num_samples, 1)               # (num_samples*B, dim+1024)
        }
        uc = {
            "crossattn": crossattn_uc.repeat(num_samples, 1, 1),
            "vector": vector_uc.repeat(num_samples, 1)
        }

        # 3. 초기 노이즈 만들기 (img2img 방식)
        base_engine.sampler.num_steps = 25

        noise = torch.randn_like(z) # random값으로 초기화
        sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).to(device)  # 각 step의 추가될 noise 만들어 놓음

        init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) / torch.sqrt(1.0 + sigmas[0]**2) # XT(완전한 gaussian noise) 생성: z + sigma*epsilon
        
        sigmas = sigmas[-img2img_timepoint:].repeat(inference_batch_size, 1)  # (inference_batch_size, steps)
        base_engine.sampler.num_steps = sigmas.shape[-1] - 1


        # 4. 샘플링 준비
        noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(
            init_z, cond=c, uc=uc, num_steps=base_engine.sampler.num_steps
        )

        with torch.cuda.amp.autocast(dtype=torch.float16):
            # 5. 디노이징 루프 XT -> X0
            for timestep in range(base_engine.sampler.num_steps):
                noised_z = base_engine.sampler.sampler_step(
                    sigmas[:, timestep],
                    sigmas[:, timestep+1],
                    lambda x, sigma, c: base_engine.denoiser(base_engine.model, x, sigma, c),
                    noised_z, cond=c, uc=uc, gamma=0
                )

        del noise, c, uc, sigmas
        torch.cuda.empty_cache()

        # 6. VAE decode
        samples_x = base_engine.decode_first_stage(noised_z) # VAE decoder 통과
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)  # [0,1] 범위로 스케일링

        enhanced_samples = torch.stack([transforms.Resize((224, 224), antialias=True)(img) for img in samples]).to(device)

        return enhanced_samples