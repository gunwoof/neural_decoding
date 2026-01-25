"""
ConnecToMind2 - Model2 Implementation (BLIP-2 ITC+ITM Style)

BLIP-2 공식 방식 완전 적용:
    [ITC - Contrastive Learning]
    - FTCLoss: fMRI-Text(CLIP) Contrastive
    - Bidirectional: fMRI→CLIP + CLIP→fMRI
    - Max pooling over query tokens
    - Learnable temperature + label smoothing
    - In-batch negatives (단일 GPU 기준)

    [ITM - Matching Classification]
    - FIMLoss: Binary classification (match/not match)
    - Classifier: nn.Linear(768, 2) - 2-class
    - Query 사용: 모든 100개 query 토큰 사용
    - Logit 평균화: mean(dim=1) - 모든 query 평균
    - Hard Negative Mining: FTC similarity 기반 torch.multinomial sampling
    - 3B Triplet: pos+pos(matched), pos+neg_clip(unmatched), neg_fmri+pos(unmatched)

커스텀 유지 (fMRI-Image Matching):
    - Attention Mask: Query+CLIP concat, use_mask=False (bidirectional)
    - Input: CLIP image embeddings [B, 257, 768] (BLIP-2의 text 역할)
    - Cross-attention: Query ← fMRI (BLIP-2의 image 역할)

Architecture (from diagram):
    fMRI [B, 100, (roi+padding)]
        -> (a) Region-level embedding -> [B, 100, 768]
        -> (b) Connectome-Q-former -> [B, 100, 768]
        -> Linear layer -> [B, 257, 768]
        -> L2 norm -> [B, 257, 768]
        -> Versatile Diffusion -> Reconstructed Image [B, 512, 512]

    Image [B, 224, 224]
        -> CLIP ViT-L/14 -> Last hidden [B, 257, 1024]
        -> Linear layer + L2 norm -> [B, 257, 768]

Loss = FIR Loss (fMRI embedding vs CLIP embedding, L1)
     + FTC Loss (BLIP-2 ITC style: bidirectional contrastive)
     + FIM Loss (BLIP-2 ITM style: 3B samples with hard negative mining)
     + Low-level Loss (L1 with target image)

Note: Low-level decoder outputs [B, 4, 32, 32] for 256x256 images.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from transformers import CLIPVisionModel, CLIPVisionModelWithProjection
from diffusers import DiffusionPipeline
from diffusers.models.autoencoder_kl import Decoder


# ============================================================================
# CLIP Image Encoder (그림의 좌측 하단) - ViT-L/14
# ============================================================================

class CLIPImageEncoder(nn.Module):
    """
    CLIP ViT-L/14 이미지 인코더

    그림 설명:
        Image -> CLIP ViT-L/14 -> Last hidden [B, 257, 1280]
                               -> Linear layer + L2 norm -> [B, 257, 768]
                               -> CLS Token [B, 768] (for Cross Entropy Loss)
    """
    def __init__(self, pretrained_model="openai/clip-vit-large-patch14", freeze=True):
        super().__init__()
        # Use CLIPVisionModelWithProjection (includes pretrained visual_projection layer)
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(pretrained_model)

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, images):
        """
        Input: images [B, 3, 224, 224]
        Output:
            hidden_state [B, 257, 768] - Pretrained projection + L2 norm
        """
        outputs = self.clip_model(images, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state  # [B, 257, 1024]

        # Post layer norm
        last_hidden = self.clip_model.vision_model.post_layernorm(last_hidden)  # [B, 257, 1024]

        # Pretrained visual projection + L2 norm (MindEye1 style)
        hidden_state = self.clip_model.visual_projection(last_hidden)  # [B, 257, 768]
        hidden_state = F.normalize(hidden_state, dim=-1)  # L2 norm

        return hidden_state


# ============================================================================
# Region-level Embedding (그림의 (a))
# ============================================================================

class RegionLevelEmbedding(nn.Module):
    """
    (a) Region-level embedding

    그림 설명:
        task-fMRI [B, 100, (roi+padding)]
        -> Flatten -> Linear projection -> [B, 100, 768]
    """
    def __init__(self, seq_len=100, input_dim=3291, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Region-level embedding: ROI별로 다른 linear layer
        self.linear_weight = nn.Parameter(torch.empty(seq_len, input_dim, embed_dim))
        for t in range(seq_len):
            init.xavier_uniform_(self.linear_weight[t])

        self.layernorm = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Input: x [B, 100, input_dim]
        Output: x [B, 100, 768]
        """
        # Ensure dtype compatibility with linear_weight
        x = x.to(dtype=self.linear_weight.dtype)

        # Region-level embedding (각 ROI별 linear)
        x = torch.einsum("btd,tdh->bth", x, self.linear_weight)  # [B, 100, 768]
        x = self.layernorm(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x


# ============================================================================
# Connectome-Q-Former (그림의 (b))
# ============================================================================

class ConnectomeQFormerBlock(nn.Module):
    """
    Connectome-Q-Former 블록: Self-attention + Cross-attention (optional) + Feed forward

    그림 설명:
        Self-attention (🔥 trainable)
        Cross-attention (🔥 trainable) - 2 layer마다 한 번
        Feed forward (🔥 trainable)
    """
    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072,
                 dropout=0.1, layer_norm_eps=1e-6):
        super().__init__()

        # Self-Attention
        self.self_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.self_drop = nn.Dropout(dropout)

        # Cross-Attention (query tokens attend to fMRI) - Query만 LayerNorm (BLIP-2 방식)
        self.cross_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_drop = nn.Dropout(dropout)

        # Feed Forward
        self.ffn_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.ffn_drop = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, fmri_feats, attn_mask=None, do_cross=True, n_q=None, cross_attn_mask=None):
        """
        Input:
            x [B, L, 768] - query tokens (+ optional CLIP hidden states)
                           L = n_q (inference) or n_q + n_t (training with mask)
            fmri_feats [B, 100, 768] - fMRI embeddings
            attn_mask: attention mask for Q-T separation
            do_cross: whether to do cross-attention in this layer
            n_q: number of query tokens (for extracting query part in cross-attention)
            cross_attn_mask: FC prior mask for cross-attention [B, n_q, seq_len]
        Output:
            x [B, L, 768]
        """
        # Self-Attention (with mask if provided)
        residual = x
        x_norm = self.self_ln(x)
        x_sa, _ = self.self_attn(x_norm, x_norm, x_norm,
                                  attn_mask=attn_mask, need_weights=False)
        x = residual + self.self_drop(x_sa)

        # Cross-Attention (query -> fMRI) - 조건부 실행, query 부분만
        if do_cross:
            q = x[:, :n_q, :]  #  Query 부분만 cross-attention 적용 -> [B, n_q, 768]
            q_res = q
            q_norm = self.cross_ln(q)
            q_ca, _ = self.cross_attn(q_norm, fmri_feats, fmri_feats,
                                     attn_mask=cross_attn_mask,  # FC prior mask 추가
                                     need_weights=False)
            q = q_res + self.cross_drop(q_ca)
            x = torch.cat([q, x[:, n_q:, :]], dim=1)  # Query + 나머지 다시 concat

        # Feed Forward
        residual = x
        x_norm = self.ffn_ln(x)
        x_ffn = self.fc2(self.ffn_drop(self.activation(self.fc1(x_norm))))
        x = residual + x_ffn  # dropout은 fc1→fc2 사이에만 적용 (BERT/GPT-2 스타일)

        return x


class ConnectomeQFormer(nn.Module):
    """
    (b) Connectome-Q-Former (initialized from CLIP ViT-L/14)

    그림 설명:
        [B, 100, 768] (fMRI) + query tokens + CLIP hidden states
        -> Connectome-Q-Former blocks (with attention mask)
        -> [B, 100, 768]

    Query tokens: 100개 (100 ROI)
    Weights initialized from openai/clip-vit-base-patch16
    Cross-attention: 2 layer마다 한 번 (BLIP-2 방식)
    Layers: 12 (CLIP ViT-B/16 기준)

    마스크 기반 Self-attention (models.py 방식):
        Query + CLIP hidden states를 concat하여 처리
        마스크로 Q-T 상호 attend 차단 -> 독립적 처리
    """
    def __init__(self, hidden_size=768, num_heads=12, num_layers=12,
                 num_query_tokens=100, dropout=0.1, cross_attention_freq=2,
                 clip_model_name="openai/clip-vit-base-patch16",
                 is_fc=False, subjects=None, fc_base_dir=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.cross_attention_freq = cross_attention_freq
        self.is_fc = is_fc
        self.num_heads = num_heads

        # Learnable FC prior scaling parameter (초기값 1.0)
        self.fc_prior_scale = nn.Parameter(torch.ones(1))

        # Cross-attention 적용 여부 미리 계산 (0, 2, 4, ... 번째 layer) - BLIP-2 방식
        self._do_cross_map = [(cross_attention_freq > 0) and (i % cross_attention_freq == 0)
                              for i in range(num_layers)]

        # Learnable query tokens [1, 100, 768] - position embedding 없음
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        # Connectome-Q-Former blocks (12 layers, CLIP ViT-B/16 기준)
        self.blocks = nn.ModuleList([
            ConnectomeQFormerBlock(hidden_size, num_heads, hidden_size * 4, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.final_ln = nn.LayerNorm(hidden_size)

        # FC prior 초기화 (subject별로 저장)
        self.fc_priors = {}  # Dict[subject_name, Tensor[100, 100]]
        if is_fc and subjects is not None and fc_base_dir is not None:
            self._load_fc_priors(subjects, fc_base_dir, num_query_tokens)

        # Initialize from CLIP weights
        self._init_from_clip(clip_model_name)

    def _init_from_clip(self, clip_model_name):
        """CLIP ViT-B/16에서 weights 가져와서 초기화 (Self-Attention + FFN)"""
        print(f"Initializing Q-Former from {clip_model_name}...")

        clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        clip_layers = clip_model.vision_model.encoder.layers

        # CLIP ViT-B/16: 12 layers, hidden_size=768 (차원 일치!)
        for i, (block, clip_layer) in enumerate(zip(self.blocks, clip_layers)):
            # Self-Attention weights
            # CLIP: layer_norm1 -> self_attn
            block.self_ln.weight.data.copy_(clip_layer.layer_norm1.weight.data)
            block.self_ln.bias.data.copy_(clip_layer.layer_norm1.bias.data)

            # HuggingFace CLIP uses separate q_proj, k_proj, v_proj
            # PyTorch MultiheadAttention uses combined in_proj_weight [3*hidden, hidden]
            # Concatenate q, k, v weights into in_proj_weight
            q_weight = clip_layer.self_attn.q_proj.weight.data
            k_weight = clip_layer.self_attn.k_proj.weight.data
            v_weight = clip_layer.self_attn.v_proj.weight.data
            block.self_attn.in_proj_weight.data.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))

            q_bias = clip_layer.self_attn.q_proj.bias.data
            k_bias = clip_layer.self_attn.k_proj.bias.data
            v_bias = clip_layer.self_attn.v_proj.bias.data
            block.self_attn.in_proj_bias.data.copy_(torch.cat([q_bias, k_bias, v_bias], dim=0))

            block.self_attn.out_proj.weight.data.copy_(clip_layer.self_attn.out_proj.weight.data)
            block.self_attn.out_proj.bias.data.copy_(clip_layer.self_attn.out_proj.bias.data)

            # FFN weights
            # CLIP: layer_norm2 -> mlp (fc1 -> activation -> fc2)
            block.ffn_ln.weight.data.copy_(clip_layer.layer_norm2.weight.data)
            block.ffn_ln.bias.data.copy_(clip_layer.layer_norm2.bias.data)
            block.fc1.weight.data.copy_(clip_layer.mlp.fc1.weight.data)
            block.fc1.bias.data.copy_(clip_layer.mlp.fc1.bias.data)
            block.fc2.weight.data.copy_(clip_layer.mlp.fc2.weight.data)
            block.fc2.bias.data.copy_(clip_layer.mlp.fc2.bias.data)

            # Cross-Attention은 랜덤 초기화 유지 (BLIP-2 방식)

        # Final LayerNorm from CLIP post_layernorm
        self.final_ln.weight.data.copy_(clip_model.vision_model.post_layernorm.weight.data)
        self.final_ln.bias.data.copy_(clip_model.vision_model.post_layernorm.bias.data)

        del clip_model
        print(f"✅ Q-Former initialized from CLIP ViT-B/16 (12 layers, hidden_size=768)")

    def _load_fc_priors(self, subjects, fc_base_dir, num_query_tokens):
        """
        모든 subject의 FC prior를 로드하고 num_heads만큼 확장하여 저장

        Args:
            subjects: list of subject names (e.g., ["sub-01", "sub-02", "sub-05", "sub-07"])
            fc_base_dir: base directory for FC prior files
            num_query_tokens: number of query tokens (100)

        FC prior shape: [100, 100] → Cross-attention mask: [num_heads, 100, 100]
            - 100개 ROI queries에 FC prior 직접 적용
            - num_heads 차원 추가: 모든 head에 동일한 FC prior 적용
        """
        import numpy as np

        seq_len = num_query_tokens  # 100 (CLS 없음)

        for sub in subjects:
            # FC prior 파일 경로
            fc_path = f"{fc_base_dir}/{sub}/{sub}_FC_schaefer-100.npy"

            if not os.path.exists(fc_path):
                print(f"⚠️  FC prior not found: {fc_path}")
                continue

            # FC prior 로드: [100, 100]
            fc_prior = np.load(fc_path).astype(np.float32)

            assert fc_prior.shape[0] == fc_prior.shape[1] == seq_len, \
                f"FC prior shape mismatch for {sub}: {fc_prior.shape} vs {seq_len}"

            # Z-score 정규화 (전체): (fc_prior - mean) / std
            # 전체 10,000개 값에 대해 하나의 분포로 정규화
            fc_mean = fc_prior.mean()  # scalar
            fc_std = fc_prior.std()    # scalar
            # std가 0이 아닐 때만 정규화
            if fc_std > 1e-6:
                fc_prior = (fc_prior - fc_mean) / fc_std

            # Cross-attention mask 생성: [100, 100] - FC prior 직접 사용
            mask = torch.from_numpy(fc_prior)  # [100, 100]

            # num_heads만큼 확장: [num_heads, 100, 100]
            # 모든 attention head에 동일한 FC prior 적용
            mask = mask.unsqueeze(0).repeat(self.num_heads, 1, 1)

            # 저장 (device 이동은 forward 시 수행)
            self.fc_priors[sub] = mask

            print(f"✓ FC prior loaded for {sub}: shape={mask.shape}, range=[{fc_prior.min():.4f}, {fc_prior.max():.4f}]")

        print(f"✅ Total {len(self.fc_priors)} FC priors loaded (pre-expanded for {self.num_heads} heads)")

    def build_mask(self, n_q, n_t, device):
        """
        Attention mask 생성 (models.py 방식)

             Q    T
        Q  [□□] [■■]
        T  [■■] [□□]

        □: attend 가능 (False)
        ■: attend 불가 (True)

        -> Q끼리만 attend, T끼리만 attend (상호 차단)
        """
        L = n_q + n_t
        mask = torch.zeros(L, L, dtype=torch.bool, device=device)
        mask[:n_q, n_q:] = True  # Q -> T 차단
        mask[n_q:, :n_q] = True  # T -> Q 차단
        return mask

    def _build_fc_prior_mask(self, subject_names, device, dtype=None):
        """
        Batch의 각 샘플에 대해 해당 subject의 FC prior mask를 생성

        Args:
            subject_names: list of subject names [B] (e.g., ["sub-01", "sub-02", ...])
            device: torch device
            dtype: torch dtype (for mixed precision compatibility)

        Returns:
            mask: [B*num_heads, 100, 100] - MultiheadAttention 형식의 FC prior mask (scaled)
        """
        masks = []
        for sub_name in subject_names:
            if sub_name in self.fc_priors:
                mask = self.fc_priors[sub_name].to(device=device, dtype=dtype if dtype else self.fc_priors[sub_name].dtype)
            else:
                # FC prior 없으면 zero mask (FC prior 없이 동작)
                mask = torch.zeros(self.num_heads, self.num_query_tokens, self.num_query_tokens,
                                 dtype=dtype if dtype else torch.float32, device=device)
            masks.append(mask)

        # [B*num_heads, 100, 100] 형태로 concat
        fc_prior_mask = torch.cat(masks, dim=0)

        # Learnable scaling parameter 적용
        fc_prior_mask = fc_prior_mask * self.fc_prior_scale

        return fc_prior_mask


    def forward(self, fmri_emb, clip_hidden=None, use_mask=True, subject_names=None):
        """
        Input:
            fmri_emb [B, 100, 768]
            clip_hidden [B, 257, 768] (optional) - CLIP hidden states for masked self-attention
            use_mask: True=Q-T 상호 차단 (FIR용), False=전범위 허용 (FIM용)
            subject_names: list of subject names [B] (for FC prior)
        Output:
            query_output [B, 100, 768]
        """
        B = fmri_emb.size(0)
        device = fmri_emb.device

        # fMRI features (position embedding 없음)
        fmri_feats = fmri_emb  # [B, 100, 768]

        # Expand query tokens for batch (position embedding 없음)
        query = self.query_tokens.expand(B, -1, -1)  # [B, 100, 768]

        # Concat query + CLIP hidden states (if provided)
        if clip_hidden is not None:
            # [B, 100, 768] + [B, 257, 768] -> [B, 357, 768]
            x = torch.cat([query, clip_hidden], dim=1)
            n_q = self.num_query_tokens
            n_t = clip_hidden.size(1)
            # use_mask=False: 전범위 허용 (FIM용) vs use_mask=True: Q-T 상호 차단 (MSE용)
            attn_mask = self.build_mask(n_q, n_t, device) if use_mask else None
        else:
            x = query
            attn_mask = None

        # FC prior mask 생성 (subject별로 동적 생성)
        cross_attn_mask = None
        if self.is_fc and subject_names is not None:
            cross_attn_mask = self._build_fc_prior_mask(subject_names, device, dtype=fmri_feats.dtype)

        # Pass through Connectome-Q-Former blocks
        for i, block in enumerate(self.blocks):
            x = block(x, fmri_feats,
                     attn_mask=attn_mask,
                     do_cross=self._do_cross_map[i],
                     n_q=self.num_query_tokens,
                     cross_attn_mask=cross_attn_mask)

        # Extract query part only (if CLIP hidden was concatenated)
        if clip_hidden is not None:
            query_output = x[:, :self.num_query_tokens, :]  # [B, 100, 768]
        else:
            query_output = x

        # Final LayerNorm
        query_output = self.final_ln(query_output)  # [B, 100, 768]

        return query_output


# ============================================================================
# Low-Level Image Decoder
# ============================================================================

class LowLevelDecoder(nn.Module):
    """
    fMRI 임베딩에서 Low-level (blurry) 이미지 생성 (MindEye2 방식)

    input:
        fmri_emb: [B, 100, 768]
    output:
        lowlevel_l1: [B, 4, 32, 32] - VAE latent space (256x256 이미지 기준)

    Shape 흐름:
        [B, 100, 768] → 76,800
            ↓ flatten
        [B, 76800] → 76,800
            ↓ Linear (압축률 18.75배)
        [B, 4096] → 64 × 8 × 8 = 4,096
            ↓ reshape
        [B, 64, 8, 8] → 4,096
            ↓ UpBlock (8→16)
        [B, 64, 16, 16] → 16,384
            ↓ UpBlock (16→32, ch: 64→4)
        [B, 4, 32, 32] → 4,096  ← Versatile Diffusion latent (256x256)
    """
    def __init__(self, seq_len=100, embed_dim=768):
        super().__init__()

        self.flatten_dim = seq_len * embed_dim  # 100 * 768 = 76800

        # fmri_emb -> [B, 64, 8, 8] feature map (64*8*8 = 4096, 압축률 18.75배)
        self.blin1 = nn.Linear(self.flatten_dim, 64 * 8 * 8, bias=True)
        self.bdropout = nn.Dropout(0.3)
        self.bnorm = nn.GroupNorm(1, 64)

        # [B, 64, 8, 8] -> [B, 4, 32, 32] (VAE latent space for 256x256, 2개 UpBlock으로 4x upsampling)
        # Note: Decoder의 마지막 block은 upsample 안 함, 그래서 block_out_channels는 len(up_block_types)+1 필요
        self.bupsampler = Decoder(
            in_channels=64,
            out_channels=4,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[64, 64, 64],  # [32, 64, 128]보다 빠름
            layers_per_block=1,
        )

    def forward(self, fmri_emb):
        """
        Input: fmri_emb [B, 100, 768]
        Output:
            lowlevel_l1: [B, 4, 32, 32] - VAE latent (for Versatile Diffusion 256x256)
        """
        B = fmri_emb.size(0)
        x = fmri_emb.view(B, -1)  # [B, 76800]

        # linear -> dropout -> reshape -> groupnorm
        lowlevel = self.blin1(x)  # [B, 4096]
        lowlevel = self.bdropout(lowlevel)
        lowlevel = lowlevel.reshape(B, 64, 8, 8).contiguous()  # [B, 64, 8, 8]
        lowlevel = self.bnorm(lowlevel)

        # VAE latent space로 upsampling (256x256 기준)
        lowlevel_l1 = self.bupsampler(lowlevel)  # [B, 4, 32, 32]

        return lowlevel_l1


# ============================================================================
# Output Projection (Linear layer + L2 norm)
# ============================================================================

class OutputProjection(nn.Module):
    """
    Q-Former 출력을 CLIP space로 projection (Transpose 방식)

    그림 설명:
        Q-Former output [B, 100, 768]
        -> Transpose -> [B, 768, 100]
        -> Linear(100, 257) -> [B, 768, 257]
        -> Transpose -> [B, 257, 768]
        -> L2 norm -> [B, 257, 768]

    파라미터 수: 100 * 257 = 25,700 (Flatten 방식 대비 ~1,200,000배 적음)
    """
    def __init__(self, input_tokens=100, output_tokens=257, hidden_size=768):
        super().__init__()
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.hidden_size = hidden_size

        # [B, 768, 100] -> [B, 768, 257] (각 dimension별 독립 projection)
        self.proj = nn.Linear(input_tokens, output_tokens)

    def forward(self, x):
        """
        Input: x [B, 100, 768]
        Output: x [B, 257, 768] (L2 normalized)
        """
        x = x.transpose(1, 2)  # [B, 768, 100]
        x = self.proj(x)       # [B, 768, 257]
        x = x.transpose(1, 2)  # [B, 257, 768]
        x = F.normalize(x, dim=-1)  # L2 norm
        return x


# ============================================================================
# Loss Functions
# ============================================================================

class FIRLoss(nn.Module):
    """
    FIR Loss (fMRI-Image Reconstruction): fMRI embedding vs CLIP embedding

    그림 설명:
        Linear layer [B, 257, 768] <------ FIR Loss ------> Linear layer [B, 257, 768]
        (from Q-Former)                                     (from CLIP)

    L1 Loss 사용: MSE보다 gradient가 안정적 (큰 오차에서도 gradient가 일정)
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, fmri_emb, clip_emb):
        """
        Input:
            fmri_emb [B, 257, 768] - Q-Former output (L2 normalized)
            clip_emb [B, 257, 768] - CLIP output (L2 normalized)
        Output: scalar loss
        """
        return self.l1(fmri_emb, clip_emb)


class FTCLoss(nn.Module):
    """
    FTC Loss (fMRI-Text(CLIP) Contrastive): BLIP-2 ITC 방식

    BLIP-2 ITC 구현:
        1. fMRI features (all tokens) vs CLIP features (CLS token) similarity 계산
        2. Max pooling over fMRI tokens (32개 query 중 최대값 선택)
        3. Bidirectional contrastive loss (fMRI→CLIP + CLIP→fMRI)
        4. In-batch negatives (단일 GPU 기준)
        5. Temperature scaling (learnable parameter)
        6. Label smoothing (0.1)

    Shapes:
        fmri_proj: [B, 257, 768] - fMRI embeddings (all tokens)
        clip_proj: [B, 257, 768] - CLIP embeddings (use CLS token [B, 768])
        sim_fmri2clip: [B, B] - fMRI→CLIP similarity (after max pooling)
        sim_clip2fmri: [B, B] - CLIP→fMRI similarity (after max pooling)
    """
    def __init__(self, temperature=0.07, label_smoothing=0.1):
        super().__init__()
        # Learnable temperature parameter (BLIP-2 style)
        self.temp = nn.Parameter(torch.ones([]) * temperature)
        self.label_smoothing = label_smoothing

    def forward(self, fmri_proj, clip_proj):
        """
        Input:
            fmri_proj [B, 257, 768] - fMRI embeddings (L2 normalized)
            clip_proj [B, 257, 768] - CLIP embeddings (L2 normalized)
        Output:
            loss_ftc: scalar
            sim_fmri2clip: [B, B] - for hard negative mining in FIM
            sim_clip2fmri: [B, B] - for hard negative mining in FIM
        """
        B = fmri_proj.size(0)
        device = fmri_proj.device

        # Normalize features
        fmri_norm = F.normalize(fmri_proj, dim=-1)  # [B, 257, 768]
        clip_cls_norm = F.normalize(clip_proj[:, 0, :], dim=-1)  # [B, 768] - CLS only

        # Compute similarity: fMRI (all tokens) vs CLIP (CLS)
        # einsum: 'bid,cd->bic' where b=fmri_batch, i=fmri_tokens, d=dim, c=clip_batch
        sim_q2t = torch.einsum('bid,cd->bic', fmri_norm, clip_cls_norm)
        # [B, 257, B] - (i-th fmri's each token) vs (all clip CLS)

        # Max over fmri tokens: [B, 257, B] -> [B, B]
        sim_fmri2clip, _ = sim_q2t.max(dim=1)  # [B, B]

        # Symmetric: CLIP (CLS) vs fMRI (all tokens)
        # einsum: 'bd,cid->bci' where b=clip_batch, d=dim, c=fmri_batch, i=fmri_tokens
        sim_t2q = torch.einsum('bd,cid->bci', clip_cls_norm, fmri_norm)
        # [B, B, 257] - (i-th clip CLS) vs (all fmri's each token)

        # Max over fmri tokens: [B, B, 257] -> [B, B]
        sim_clip2fmri, _ = sim_t2q.max(dim=-1)  # [B, B]

        # Temperature scaling (learnable)
        sim_fmri2clip = sim_fmri2clip / self.temp
        sim_clip2fmri = sim_clip2fmri / self.temp

        # Targets: diagonal (positive pairs)
        targets = torch.arange(B, device=device, dtype=torch.long)

        # Bidirectional contrastive loss (BLIP-2 ITC style)
        loss_ftc = (
            F.cross_entropy(sim_fmri2clip, targets, label_smoothing=self.label_smoothing)
            + F.cross_entropy(sim_clip2fmri, targets, label_smoothing=self.label_smoothing)
        ) / 2

        return loss_ftc, sim_fmri2clip, sim_clip2fmri


class FIMLoss(nn.Module):
    """
    FIM Loss (fMRI-Image Matching): BLIP-2 ITM 방식 with Hard Negative Mining

    BLIP-2 ITM 구현:
        1. ITC similarity를 사용하여 hard negative sampling
        2. 3B triplet 구성: (pos+pos, pos+neg_clip, neg_fmri+pos)
        3. Q-Former forward with 3B samples (use_mask=False)
        4. Binary classification (match=1, not match=0)
        5. All query tokens 사용 + mean pooling

    Hard Negative Mining:
        - sim_fmri2clip, sim_clip2fmri에서 diagonal 제외
        - Softmax로 확률 분포 변환
        - torch.multinomial로 hard negative sampling

    Shapes:
        fmri_emb: [B, 100, 768] - Region-level embeddings
        clip_proj: [B, 257, 768] - CLIP embeddings
        sim_fmri2clip: [B, B] - fMRI→CLIP similarity (from FTC)
        sim_clip2fmri: [B, B] - CLIP→fMRI similarity (from FTC)
        Output: scalar loss
    """
    def __init__(self, qformer, fim_classifier):
        super().__init__()
        self.qformer = qformer
        self.fim_classifier = fim_classifier

    def forward(self, fmri_emb, clip_proj, sim_fmri2clip, sim_clip2fmri, subject_names=None):
        """
        Input:
            fmri_emb [B, 100, 768] - Region-level embeddings
            clip_proj [B, 257, 768] - CLIP embeddings
            sim_fmri2clip [B, B] - fMRI→CLIP similarity (from FTC)
            sim_clip2fmri [B, B] - CLIP→fMRI similarity (from FTC)
            subject_names: list of subject names [B] (for FC prior)
        Output:
            loss_fim: scalar
        """
        B = fmri_emb.size(0)
        device = fmri_emb.device

        # Step 1: Hard negative mining (BLIP-2 style: within torch.no_grad)
        with torch.no_grad():
            # Mask diagonal (positive pairs)
            sim_fmri2clip_neg = sim_fmri2clip.clone()
            sim_clip2fmri_neg = sim_clip2fmri.clone()
            sim_fmri2clip_neg.fill_diagonal_(-10000)
            sim_clip2fmri_neg.fill_diagonal_(-10000)

            # Softmax to get sampling weights
            weights_fmri2clip = F.softmax(sim_fmri2clip_neg, dim=1)  # [B, B]
            weights_clip2fmri = F.softmax(sim_clip2fmri_neg, dim=1)  # [B, B]

        # Step 2: Sample hard negatives with torch.multinomial
        # Hard negative clip for each fmri
        clip_neg_indices = []
        for b in range(B):
            neg_idx = torch.multinomial(weights_fmri2clip[b], 1).item()
            clip_neg_indices.append(neg_idx)
        clip_neg_indices = torch.tensor(clip_neg_indices, device=device, dtype=torch.long)
        clip_proj_neg = clip_proj[clip_neg_indices]  # [B, 257, 768]

        # Hard negative fmri for each clip
        fmri_neg_indices = []
        for b in range(B):
            neg_idx = torch.multinomial(weights_clip2fmri[b], 1).item()
            fmri_neg_indices.append(neg_idx)
        fmri_neg_indices = torch.tensor(fmri_neg_indices, device=device, dtype=torch.long)
        fmri_emb_neg = fmri_emb[fmri_neg_indices]  # [B, 100, 768]

        # Step 4: Construct 3B samples (BLIP-2 triplet style)
        # 1) fmri(pos) + clip(pos)  → matched (label=1)
        # 2) fmri(pos) + clip(neg)  → unmatched (label=0)
        # 3) fmri(neg) + clip(pos)  → unmatched (label=0)
        fmri_emb_3b = torch.cat([
            fmri_emb,      # [B, 100, 768] - positive
            fmri_emb,      # [B, 100, 768] - positive (for negative clip)
            fmri_emb_neg,  # [B, 100, 768] - negative fmri
        ], dim=0)  # [3B, 100, 768]

        clip_proj_3b = torch.cat([
            clip_proj,      # [B, 257, 768] - positive
            clip_proj_neg,  # [B, 257, 768] - negative clip
            clip_proj,      # [B, 257, 768] - positive (for negative fmri)
        ], dim=0)  # [3B, 257, 768]

        fim_labels = torch.cat([
            torch.ones(B, device=device),   # [B] - matched
            torch.zeros(2 * B, device=device),  # [2B] - unmatched
        ], dim=0)  # [3B]

        # Subject names 3배로 복제
        if subject_names is not None:
            subject_names_neg = [subject_names[i] for i in fmri_neg_indices.cpu().tolist()]
            subject_names_3b = subject_names + subject_names + subject_names_neg
        else:
            subject_names_3b = None

        # Step 5: Q-Former forward with 3B samples (use_mask=False)
        qformer_out_fim = self.qformer(
            fmri_emb_3b, clip_proj_3b, use_mask=False, subject_names=subject_names_3b
        )  # [3B, 100, 768]

        # Step 6: Binary classification (BLIP-2 ITM style)
        vl_embeddings = qformer_out_fim  # [3B, 100, 768] - 모든 query 사용
        vl_output = self.fim_classifier(vl_embeddings)  # [3B, 100, 2] - 각 query에서 2-class
        fim_logits = vl_output.mean(dim=1)  # [3B, 2] - 모든 query 평균화

        # Cross entropy loss (2-class)
        loss_fim = F.cross_entropy(fim_logits, fim_labels.long())

        return loss_fim


# ============================================================================
# Complete Model (BLIP-2 ITM Style)
# ============================================================================

class ConnecToMind2(nn.Module):
    """
    ConnecToMind2 - Model2 (BLIP-2 ITC+ITM Style)

    BLIP-2 방식 완전 적용:
        [FTCLoss - Contrastive Learning (ITC 기반)]
        - Bidirectional contrastive loss (fMRI→CLIP + CLIP→fMRI)
        - Max pooling over query tokens
        - Learnable temperature + label smoothing
        - In-batch negatives (단일 GPU 기준)

        [FIMLoss - Matching Classification (ITM 기반)]
        - Classifier: nn.Linear(768, 2) - 2-class
        - Query 사용: 모든 100개 query 토큰 사용
        - Logit 평균화: mean(dim=1) - 모든 query 평균
        - Hard Negative Mining: FTC similarity 기반 sampling
        - 3B Triplet: pos+pos(1), pos+neg_clip(0), neg_fmri+pos(0)

    Architecture:
        fMRI [B, 100, input_dim]
            -> (a) Region-level embedding -> [B, 100, 768]
            -> (b) Connectome-Q-Former -> [B, 100, 768]
            -> Linear layer -> [B, 257, 768]
            -> L2 norm -> [B, 257, 768]

        Image [B, 3, 224, 224]
            -> CLIP ViT-L/14 -> [B, 257, 1024]
            -> Linear layer + L2 norm -> [B, 257, 768]

    Loss = FIR Loss (fMRI embedding vs CLIP embedding, L1)
         + FTC Loss (BLIP-2 ITC style: bidirectional contrastive)
         + FIM Loss (BLIP-2 ITM style: 3B samples with hard negatives)
         + Low-level Loss (VAE L1)

    Output (training):
        fmri_proj: [B, 257, 768] - Q-Former output (Linear + L2 norm)
        clip_proj: [B, 257, 768] - CLIP output (Linear + L2 norm)
        lowlevel_l1: [B, 4, 32, 32] - VAE latent (for Versatile Diffusion 512x512)
        loss_fir: FIR loss
        loss_ftc: FTC loss (ITC-based contrastive)
        loss_fim: FIM loss (3B samples, hard negative mining)

    Versatile Diffusion inputs:
        - image_embeds: fmri_proj [B, 257, 768]
        - image: VAE decode(lowlevel_l1) for image condition (256x256)
    """
    def __init__(self, seq_len=100, input_dim=3291, embed_dim=768,
                 num_qformer_layers=12, num_query_tokens=100,
                 is_fc=False, subjects=None, fc_base_dir=None):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_query_tokens = num_query_tokens

        # 1. CLIP Image Encoder (ViT-L/14, frozen)
        self.clip_encoder = CLIPImageEncoder(freeze=True)

        # 2. Region-level Embedding
        self.region_embedding = RegionLevelEmbedding(
            seq_len=seq_len,
            input_dim=input_dim,
            embed_dim=embed_dim
        )

        # 3. Connectome-Q-Former (initialized from CLIP, with FC prior)
        self.connectome_qformer = ConnectomeQFormer(
            hidden_size=embed_dim,
            num_heads=12,
            num_layers=num_qformer_layers,
            num_query_tokens=num_query_tokens,
            dropout=0.1,
            is_fc=is_fc,
            subjects=subjects,
            fc_base_dir=fc_base_dir
        )

        # 4. Output Projection: [B, 100, 768] -> [B, 257, 768]
        self.output_proj = OutputProjection(
            input_tokens=num_query_tokens,
            output_tokens=257,
            hidden_size=embed_dim
        )

        # 5. Low-Level Decoder
        self.low_level_decoder = LowLevelDecoder(seq_len=seq_len, embed_dim=embed_dim)

        # 6. FIM classifier (BLIP-2 ITM Style): 모든 query [768] -> 2-class logit [2]
        self.fim_classifier = nn.Linear(embed_dim, 2)

        # 7. Loss functions
        self.fir_loss_fn = FIRLoss()
        self.ftc_loss_fn = FTCLoss(temperature=0.07, label_smoothing=0.1)
        self.fim_loss_fn = FIMLoss(qformer=self.connectome_qformer, fim_classifier=self.fim_classifier)

    def forward(self, fmri, images, device, subject_names=None):
        """
        Training forward (Q-Former 2번 호출: FIR용 마스크 O, FIM용 마스크 X)

        FIR: Q-T 상호 차단 마스크 사용 (Query가 CLIP을 직접 보면 cheating)
        FTC: ITC 기반 contrastive learning (fMRI-CLIP alignment)
        FIM: 마스크 없음 (Query가 CLIP을 보고 matching 판단)
             Hard Negative Mining + 3B samples (BLIP-2 ITM 공식 방식)

        Loss Classes:
            - FIRLoss: L1 loss (fmri_proj vs clip_proj)
            - FTCLoss: Bidirectional contrastive loss (ITC 방식)
            - FIMLoss: Binary classification with hard negative mining (ITM 방식)

        Input:
            fmri [B, 100, input_dim]
            images [B, 3, 224, 224]
            device: torch device
            subject_names: list of subject names [B] (for FC prior)

        Output: dict
            fmri_proj: [B, 257, 768] - for FIR loss
            clip_proj: [B, 257, 768] - for FIR loss
            lowlevel_l1: [B, 4, 32, 32] - for Versatile Diffusion 512x512
            loss_fir: scalar
            loss_ftc: scalar (ITC-based contrastive)
            loss_fim: scalar (BLIP-2 ITM style with 3B samples)
        """
        B = fmri.size(0)

        # === Image path (CLIP) ===
        clip_proj = self.clip_encoder(images)  # [B, 257, 768]

        # === fMRI path ===
        # (a) Region-level embedding
        fmri_emb = self.region_embedding(fmri)  # [B, 100, 768]

        # === FIR Branch (마스크 O: Q-T 상호 차단) ===
        qformer_out_fir = self.connectome_qformer(fmri_emb, clip_proj, use_mask=True, subject_names=subject_names)  # [B, 100, 768]
        fmri_proj = self.output_proj(qformer_out_fir)  # [B, 257, 768]

        # === Compute Losses ===

        # 1. FIR Loss (L1)
        loss_fir = self.fir_loss_fn(fmri_proj, clip_proj)

        # 2. FTC Loss (ITC-based Contrastive) - returns similarity matrices for hard negative mining
        loss_ftc, sim_fmri2clip, sim_clip2fmri = self.ftc_loss_fn(fmri_proj, clip_proj)

        # 3. FIM Loss (ITM with Hard Negative Mining)
        loss_fim = self.fim_loss_fn(fmri_emb, clip_proj, sim_fmri2clip, sim_clip2fmri, subject_names=subject_names)

        # 4. Low-level decoder (원본 fmri_emb만 사용)
        lowlevel_l1 = self.low_level_decoder(fmri_emb)  # [B, 4, 64, 64]

        return {
            "fmri_proj": fmri_proj,          # [B, 257, 768]
            "clip_proj": clip_proj,          # [B, 257, 768]
            "lowlevel_l1": lowlevel_l1,      # [B, 4, 64, 64]
            "loss_fir": loss_fir,
            "loss_ftc": loss_ftc,
            "loss_fim": loss_fim,
        }

    def inference(self, fmri, subject_names=None):
        """
        Inference (이미지 없이)

        Input:
            fmri [B, 100, input_dim]
            subject_names: list of subject names [B] (for FC prior)
        Output:
            fmri_proj: [B, 257, 768] - Versatile Diffusion의 image_embeds로 사용
            lowlevel_l1: [B, 4, 32, 32] - Versatile Diffusion latent (512x512)
        """
        # (a) Region-level embedding
        fmri_emb = self.region_embedding(fmri)  # [B, 100, 768]

        # (b) Connectome-Q-Former
        qformer_out = self.connectome_qformer(fmri_emb, subject_names=subject_names)  # [B, 100, 768]

        # Linear layer + L2 norm
        fmri_proj = self.output_proj(qformer_out)  # [B, 257, 768]

        # Low-level decoder
        lowlevel_l1 = self.low_level_decoder(fmri_emb)  # [B, 4, 64, 64]

        return {
            "fmri_proj": fmri_proj,      # [B, 257, 768]
            "lowlevel_l1": lowlevel_l1,  # [B, 4, 64, 64]
        }


# ============================================================================
# Model Factory
# ============================================================================

def get_model(args):
    """
    모델 생성

    args 필요 속성:
        - seq_len, input_dim, embed_dim, num_qformer_layers, num_query_tokens
        - cache_dir: pretrained model cache 경로

    returns:
        connectomind2: 메인 모델
        versatile_diffusion: Versatile Diffusion pipeline
        vae: VAE encoder (for L1 loss)
        l1: L1 loss function
    """
    # 캐시 경로 (로컬 우선, 없으면 온라인에서 다운로드)
    cache_dir = args.cache_dir

    # FC prior base directory (is_fc가 True인 경우에만 사용)
    fc_base_dir = None
    if args.is_fc:
        fc_base_dir = f"{args.root_dir}/{args.fmri_dir}/{args.fmri_detail_dir}"

    # 메인 모델
    connectomind2 = ConnecToMind2(
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        num_qformer_layers=args.num_qformer_layers,
        num_query_tokens=args.num_query_tokens,
        is_fc=args.is_fc,
        subjects=args.subjects,
        fc_base_dir=fc_base_dir,
    )

    # High-level reconstruction 용도 -> Versatile Diffusion pipeline
    # 로컬에 있으면 로컬에서, 없으면 온라인에서 다운로드 후 cache_dir에 저장
    print("Loading Versatile Diffusion pipeline...")
    try:
        versatile_diffusion = DiffusionPipeline.from_pretrained(
            "shi-labs/versatile-diffusion",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            local_files_only=True  # 로컬 우선 시도
        )
        print("✅ Versatile Diffusion loaded (from local cache)")
    except Exception:
        print("  로컬 캐시 없음, 온라인에서 다운로드...")
        try:
            versatile_diffusion = DiffusionPipeline.from_pretrained(
                "shi-labs/versatile-diffusion",
                torch_dtype=torch.float16,
                cache_dir=cache_dir
            )
            print("✅ Versatile Diffusion downloaded and cached")
        except Exception as e:
            print(f"[!] Versatile Diffusion 로딩 실패: {e}")
            versatile_diffusion = None

    # Low-level reconstruction 용도 -> VAE (for L1 loss - low-level)
    print("Loading VAE...")
    try:
        sd_pipe = DiffusionPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            cache_dir=cache_dir,
            local_files_only=True  # 로컬 우선 시도
        )
        print("✅ VAE loaded (from local cache)")
    except Exception:
        print("  로컬 캐시 없음, 온라인에서 다운로드...")
        sd_pipe = DiffusionPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            cache_dir=cache_dir
        )
        print("✅ VAE downloaded and cached")
    vae = sd_pipe.vae
    vae.eval().requires_grad_(False)

    # L1 loss
    l1 = nn.L1Loss()

    return {
        "connectomind2": connectomind2,
        "versatile_diffusion": versatile_diffusion,
        "vae": vae,
        "l1": l1,
    }
