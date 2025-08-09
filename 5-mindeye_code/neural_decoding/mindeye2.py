import os
import torch
import torch.nn as nn
from torchvision import transforms
from omegaconf import OmegaConf

# for high-level image 
from pretrained_cache.generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # ViT/bigG

# for low-level
from diffusers import DiffusionPipeline # vae encoder & decoder 용도
from diffusers.models.autoencoder_kl import Decoder # upsampling 용도
from convnext import ConvnextXL # loss 용도

# for prior
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists, RotaryEmbedding, CausalTransformer, SinusoidalPosEmb, MLP, Rearrange, repeat, rearrange, prob_mask_like, LayerNorm, RelPosBias, Attention, FeedForward

# for caption
from pretrained_cache.BrainCaptioning.modeling_git import GitForCausalLMClipEmb # caption token 생성
from transformers import AutoProcessor # (caption token -> english) 변환
from pretrained_cache.generative_models.sgm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2 # SDXL이 text condition을 2개의 embedding으로 받음

# for SDXL
from pretrained_cache.generative_models.sgm.models.diffusion import DiffusionEngine

# for aug
import kornia
from kornia.augmentation.container import AugmentationSequential

# Shared-subject latent space
class RidgeRegression(nn.Module):
    # make sure to add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_sizes, out_features=4096): 
        super().__init__()

        # subject마다 voxel개수가 달라서 각각 linear layer를 사용해야함
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
        
    def forward(self, x, subj_idx):
        '''
            out: - shape: [batch, 4096]
        '''
        out = self.linears[subj_idx](x).unsqueeze(1) # MLP-Mixer를 사용하기 위해 [B, 4096] -> [B, 1, 4096]
        return out

# Difusion prior재료 + Retrieval 재료 + low-level 재료
class BrainNetwork(nn.Module):
    def __init__(self, h=4096, in_dim=4096, out_dim=256*1664, seq_len=1, n_blocks=4, drop=.15, clip_size=1664):
        super().__init__()
        self.h = h
        self.seq_len = seq_len
        self.clip_size = clip_size
        self.backbone_mlp = nn.ModuleList([
            self.mlp(h, h, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)
        
        # low-level submodule
        # self.blin1 = nn.Linear(h*seq_len,4*28*28,bias=True) # [4, 64, 64]을 만들기 위해 버림
        self.blin1 = nn.Linear(h*seq_len,64*8*8,bias=True)
        self.bdropout = nn.Dropout(.3)
        self.bnorm = nn.GroupNorm(1, 64)
        self.bupsampler = Decoder(
            in_channels=64,
            out_channels=4,
            up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
            block_out_channels=[32, 64, 128],
            layers_per_block=1,
        )
        self.b_maps_projector = nn.Sequential(
            nn.Conv2d(64, 512, 1, bias=False),
            nn.GroupNorm(1,512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=False),
            nn.GroupNorm(1,512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=True),
        )

    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )

    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )
    
    def forward(self, x):
        '''
            x(Backbone): - shape: [batch, (256*1664)]
            x(projector): - shape: [batch, (256*1664)]
            x(low-level): - shape: ([batch, 4, 64, 64], [batch, 64, 512])
        '''
        # Residual MLP backbone
        residual = x
        for block in self.backbone_mlp:
            x = block(x) + residual
            residual = x
        x = x.reshape(x.size(0), -1) # [B, 4096]
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size) # [B, 4096] -> [B, 256*1664] -> [B, 256, 1664]

        # make empty tensors
        retrieval,lowlevel = torch.Tensor([0.]), torch.Tensor([0.])

        # retrieval submodule
        retrieval = self.clip_proj(backbone) # [B, 256, 1664]

        # low-level submodule
        lowlevels = (None, None)
        if self.blurry_recon:
            lowlevel = self.blin1(x) # [B, 64*8*8]
            lowlevel = self.bdropout(lowlevel)
            lowlevel = lowlevel.reshape(lowlevel.shape[0], -1, 8, 8).contiguous() # [B, 64, 8, 8]
            lowlevel = self.bnorm(lowlevel) 

            # l1 loss에 사용됨 -> 실제 blurry image prediction 값
            lowlevel_l1 = self.bupsampler(lowlevel)

            # ConvNext loss에 사용됨
            lowlevel_aux = self.b_maps_projector(lowlevel).flatten(2).permute(0,2,1) # [B, 64, 512]

            lowlevels = (lowlevel_l1, lowlevel_aux) # ([B, 4, 64, 64], [B, 64, 512])
        
        # backbone(Diffusion prior), retrieval(retrieval submodule), lowlevel(low-level submodule)
        return backbone, retrieval, lowlevels

# PriorNetwork(1 step)가 인자로 들어감  
class BrainDiffusionPrior(DiffusionPrior):
    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None) # 부모(DiffusionPrior)는 voxel2clip을 모르기 때문에 빼놓음
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    def forward(self, text_embed = None, image_embed = None, *args, **kwargs):
        '''
        loss: loss(prediction x_0, target x_0) - shape: Scala
        pred: x_t -> x_t-1 denoise한 결과 - shape: [batch, 256, 1664]
        '''
        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch) # backward할 때 random으로 시간 뽑음

        # ex) ex: 1.0 ~ 2.5 -> prediction 후 self.image_embed_scale을 나눠줘야 함
        image_embed *= self.image_embed_scale

        # calculate forward loss
        loss, pred = self.p_losses(image_embed=image_embed, times=times, text_embed=text_embed, *args, **kwargs)
        
        # undo the scaling so we can directly use it for real mse loss and reconstruction
        return loss, pred

    def p_losses(self, image_embed, times, text_embed=None, noise = None):
        '''
        loss: loss(prediction x_0, target x_0) - shape: Scala
        pred: x_t -> x_t-1 denoise한 결과 - shape: [batch, 256, 1664]
        '''
        # noise 정의
        noise = default(noise, lambda: torch.randn_like(image_embed)) 

        # random한 t가 들어옴 -> x_0에서 한 번에 t까지의 noise를 씌움 
        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        # prediction 값(x_0)
        pred = self.net(image_embed_noisy, times, text_embed=text_embed) # cond_drop_prob: 텍스트 & 이미지 드롭아웃 확률 

        # prediction 값(x_0) normalization
        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        # x_0를 비교하는 방식
        target = image_embed

        loss = nn.functional.mse_loss(pred, target) # diffusion prior에서는 ε가 아닌 x_0를 예측함
        return loss, pred # train 할 때 보통 loss와 prediction 같이 반환

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps = None, **kwargs):

        image_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps = timesteps)

        return image_embed
    
    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, self_cond = None, clip_denoised = True, cond_scale = 1.):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, text_cond = text_cond, self_cond = self_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)

        noise = torch.randn_like(x)

        # t-1에 denoise 
        # 생성
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # 적용
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

# dalle2의 diffusionprior에서 x0(보통은 입실론)을 뽑음 only 1 step
class PriorNetwork(nn.Module):
    def __init__(
        self,
        dim, # 1664
        num_timesteps = None, # time step 수 ex) 1000
        num_time_embeds = 1, # time embedding 개수
        num_tokens = 256,
        causal = True,
        learned_query_mode = 'none', 
        **kwargs
    ):
        super().__init__()
        self.dim = dim 
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        # time embedding으로 변환
        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # query(pure noise로 만들어짐)
        if self.learned_query_mode == 'pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)

        # transformer 저장
        self.causal_transformer = FlaggedCausalTransformer(dim = dim, causal=causal, **kwargs)


        # tokens.shape = [batch_size, total_token, dim]
        # total_token= [brain_embed(256개), time_embed(1개), image_embed(256개)] -> 이 중에서 맞춰야 할 embed
        self.num_tokens = num_tokens

    def forward(
        self,
        image_embed, # pure noise 
        diffusion_timesteps,
        *,
        text_embed=None,
    ):  
        # brain embed = text embed  
        if text_embed is not None:
            brain_embed = text_embed 
        
        batch, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype

        # time_embedding
        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        # token 정의
        if self.learned_query_mode == 'pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch) # repeat: '257 768' shape -> 'b 257 768' shape로 변경
            image_embed = image_embed + pos_embs # pure noise로 만들어짐 -(forward)-> prediction 값

            tokens = torch.cat((
                brain_embed,  # [b, 257, 768] 
                time_embed,  # [b, 1, 768]
                image_embed,  # [b, 257, 768]
            ), dim = -2) # [b, 257 + 1 + 257 + 257, 768] = [b, 515, 768]

        # transformer
        tokens = self.causal_transformer(tokens) # output: [b, 515, 768]

        pred_image_embed = tokens[..., -self.num_tokens:, :] # output: [b, 257, 768] - image_embed를 prediction값으로 사용

        return pred_image_embed

class FlaggedCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth, # transformer의 layer 수 -> versatile의 한 step의 layer
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True,
        causal=True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)

# CLIP ViT/bigG-14[batch, 256, 1664] -> CLIP ViT/L-14[batch, 257, 1024]
class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 257)
        self.linear2 = nn.Linear(1664, 1024)
    def forward(self, x): 
        '''
            x: - shape: [batch, 257, 1024]
        '''
        x = x.permute(0,2,1) 
        x = self.linear1(x)
        x = self.linear2(x.permute(0,2,1))
        return x

# 학습하는 모델을 담을 class
class MindEyeModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
def get_model(args):

    #### image clip 정의 ####
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )

    #### 학습하는 모델을 담을 객체 ####
    mindeye2 = MindEyeModule()
        
    #### ridge(Shared-subject latent space) 정의 ####
    num_voxels_list = [15724, 14278, 13039, 12682] # subject 1,2,5,7
    mindeye2.ridge = RidgeRegression(num_voxels_list=num_voxels_list, out_features=4096).to(args.device)

    #### Residual MLP backbone 정의 ####
    mindeye2.backbone = BrainNetwork(h=4096, in_dim=4096, out_dim=256*1664, seq_len=1, n_blocks=4, clip_size=1664).to(args.device)

    #### difussion prior 정의 ####
    clip_seq_dim = 256
    out_dim = 1664 # 모델 거치면 256 * 1664 나옴
    depth = 6 # transformer의 layer 수 -> difussion prior의 한 step의 layer
    dim_head = 52 # head당 dim
    heads = out_dim//dim_head # attention head 수
    timesteps = 100 # difusion step 수 

    prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = clip_seq_dim,
            learned_query_mode="pos_emb"
    ).to(args.device)

    mindeye2.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    ).to(args.device)

    #### low-level 정의 ####
    #### encoder & decoder 정의 ####
    try:
        # 전체 pipeline 로컬에서 로드
        sd_model_dir = os.path.join(args.cache_dir, "models--lambdalabs--sd-image-variations-diffusers", "snapshots")
        snapshot_name = os.listdir(sd_model_dir)  # 첫 snapshot
        snapshot_path = os.path.join(sd_model_dir, snapshot_name[0])

        sd_pipe = DiffusionPipeline.from_pretrained(snapshot_path)
    except Exception as e:
        print(f"[!] 로컬 snapshot 로딩 실패, 온라인에서 전체 모델 로드: {e}")
        sd_pipe = DiffusionPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", cache_dir=args.cache_dir)

    # 학습 비활성화
    sd_pipe.vae.eval().requires_grad_(False)
    
    vae = sd_pipe.vae
    noise_scheduler = sd_pipe.scheduler

    # convnext + mlp 정의(loss에서만 사용)
    cnx_path = os.path.join(args.cache_dir, "convnext_xlarge_alpha0.75_fullckpt.pth")
    cnx = ConvnextXL(cnx_path)
    cnx.requires_grad_(False)
    cnx.eval()

    # L1 loss
    l1 = nn.L1Loss()

    #### caption 정의 ####
    clip_linear = CLIPConverter() # CLIP ViT/bigG-14[batch, 256, 1664] -> CLIP ViT/L-14[batch, 257, 1024]
    clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco") # caption token 생성
    token_to_text = AutoProcessor.from_pretrained("microsoft/git-large-coco") # (caption token -> english) 변환

    #### reconstruction 정의 ####
    # config 정의
    config = OmegaConf.load("pretrained_cache/generative_models/configs/inference/sd_xl_base.yaml") # yaml불러오기
    config = OmegaConf.to_container(config, resolve=True) # yaml dic으로 변환
    refiner_params = config["model"]["params"]
    network_config = refiner_params["network_config"]
    denoiser_config = refiner_params["denoiser_config"]
    first_stage_config = refiner_params["first_stage_config"]
    conditioner_config = refiner_params["conditioner_config"]
    scale_factor = refiner_params["scale_factor"]
    disable_first_stage_autocast = refiner_params["disable_first_stage_autocast"]

    config = OmegaConf.load("generative_models/configs/unclip6.yaml") # yaml불러오기
    config = OmegaConf.to_container(config, resolve=True) # yaml dic으로 변환
    unclip_params = config["model"]["params"]
    sampler_config = unclip_params["sampler_config"]
    sampler_config['params']['num_steps'] = 38

    # caption을 text embedding으로 변환
    base_text_embedder1 = FrozenCLIPEmbedder(
    layer=conditioner_config['params']['emb_models'][0]['params']['layer'],
    layer_idx=conditioner_config['params']['emb_models'][0]['params']['layer_idx'],
    )
    base_text_embedder2 = FrozenOpenCLIPEmbedder2(
        arch=conditioner_config['params']['emb_models'][1]['params']['arch'],
        version=conditioner_config['params']['emb_models'][1]['params']['version'],
        freeze=conditioner_config['params']['emb_models'][1]['params']['freeze'],
        layer=conditioner_config['params']['emb_models'][1]['params']['layer'],
        always_return_pooled=conditioner_config['params']['emb_models'][1]['params']['always_return_pooled'],
        legacy=conditioner_config['params']['emb_models'][1]['params']['legacy'],
    )

    # sdxl 정의
    sdxl_path = os.path.join(args.cache_dir, "generative_models", "zavychromaxl_v30.safetensors")
    sdxl = DiffusionEngine(network_config=network_config,
                       denoiser_config=denoiser_config,
                       first_stage_config=first_stage_config,
                       conditioner_config=conditioner_config,
                       sampler_config=sampler_config, # using the one defined by the unclip
                       scale_factor=scale_factor,
                       disable_first_stage_autocast=disable_first_stage_autocast,
                       ckpt_path=sdxl_path)
    sdxl.eval().requires_grad_(False)

    # sdxl unclip 정의
    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    sdxl_unclip_path = os.path.join(args.cache_dir, "generative_models", "unclip6_epoch0_step110000.ckpt")
    sdxl_unclip = DiffusionEngine(network_config=network_config,
                       denoiser_config=denoiser_config,
                       first_stage_config=first_stage_config,
                       conditioner_config=conditioner_config,
                       sampler_config=sampler_config,
                       scale_factor=scale_factor,
                       disable_first_stage_autocast=disable_first_stage_autocast)
    ckpt = torch.load(sdxl_unclip_path, map_location=args.device)
    sdxl_unclip.load_state_dict(ckpt['state_dict'])
    sdxl_unclip.eval().requires_grad_(False)

    models = {
        "clip": clip_img_embedder,
        "mindeye2": mindeye2,
        "vae": vae, 
        "noise_scheduler": noise_scheduler, 
        "cnx": cnx,
        "l1": l1,
        "clip_linear": clip_linear,
        "clip_text_model": clip_text_model,
        "token_to_text": token_to_text,
        "base_text_embedder1": base_text_embedder1,
        "base_text_embedder2": base_text_embedder2,
        "sdxl": sdxl,
        "sdxl_unclip": sdxl_unclip
    }

    return models



