"""
ConnecToMind2 - Complete Implementation

Architecture (from diagram):
    fMRI [B, 20, (roi+padding)]
        -> (a) Region-level embedding -> [B, 20, 1280]
        -> (b) Connectome-Transformer -> [B, 20, 1280]
        -> (c) Q-Former with fMRI -> [B, 1024] (Pool + L2 norm)
        -> Stable Diffusion 2.1-unclip -> Reconstructed Image

    Image [B, 3, 224, 224]
        -> CLIP ViT-H -> Last hidden [B, 257, 1280]
                      -> CLS 뽑기 [B, 1280]
                      -> linear layer + L2 norm -> [B, 1024]

Loss = fMRI-image contrastive learning + fMRI-image matching + low-level image MSE
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import gc

from transformers import CLIPVisionModel
from diffusers import StableUnCLIPImg2ImgPipeline

from diffusers import DiffusionPipeline
from convnext import ConvnextXL
from diffusers.models.autoencoder_kl import Decoder

# ============================================================================
# CLIP Image Encoder (그림의 좌측 하단)
# ============================================================================

class CLIPImageEncoder(nn.Module):
    """
    CLIP ViT-H/14 이미지 인코더

    그림 설명:
        Image -> CLIP ViT-H -> Last hidden [B, 257, 1280]
                            -> CLS 뽑기 [B, 1280]
                            -> linear layer + L2 norm -> [B, 1024]
    """
    def __init__(self, pretrained_model="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", freeze=True):
        super().__init__()
        self.clip_model = CLIPVisionModel.from_pretrained(pretrained_model)

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # linear layer: CLS [B, 1280] -> [B, 1024]
        self.proj = nn.Linear(1280, 1024)

    def forward(self, images):
        """
        Input: images [B, 3, 224, 224]
        Output:
            hidden_state [B, 257, 1280] - Q-Former Self-attention 입력
            image_emb [B, 1024] - FIC Loss 타겟 (linear + L2 norm)
        """
        outputs = self.clip_model(images, output_hidden_states=True)
        hidden_state = outputs.last_hidden_state  # [B, 257, 1280]
        cls_token = hidden_state[:, 0, :]  # CLS 뽑기 [B, 1280]
        image_emb = self.proj(cls_token)  # linear layer [B, 1024]
        image_emb = F.normalize(image_emb, dim=-1)  # L2 norm
        return hidden_state, image_emb


# ============================================================================
# Connectome-Transformer (그림의 (a), (b))
# ============================================================================

class ConnectomeTransformer(nn.Module):
    """
    (a) Region-level embedding + (b) Connectome-Transformer

    그림 설명:
        task-fMRI [B, 20, (roi+padding)]
        -> Flatten -> (a) Region-level embedding [B, 20, 1280]
        -> Positional Embedding + Feed forward
        -> (b) Connectome-Transformer [B, 20, 1280]
    """
    def __init__(self, seq_len=20, input_dim=2056, embed_dim=1280, nhead=8, num_layers=8,
                 is_fc=False, fc_matrix_path=""):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.is_fc = is_fc

        # (a) Region-level embedding: ROI별로 다른 linear layer
        self.linear1_weight = nn.Parameter(torch.empty(seq_len, input_dim, embed_dim))
        for t in range(seq_len):
            init.xavier_uniform_(self.linear1_weight[t])
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # (b) Connectome-Transformer
        if is_fc:
            encoder_layer = CustomTransformerEncoderLayer(fc_matrix_path=fc_matrix_path, d_model=embed_dim, nhead=nhead)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Input: x [B, 20, 2056]
        Output: x [B, 20, 1280]
        """
        # (a) Region-level embedding
        x = torch.einsum("btd,tdh->bth", x, self.linear1_weight)  # [B, 20, 1280]
        x = self.layernorm1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        # Positional embedding
        x = x + self.pos_embedding

        # (b) Connectome-Transformer (Feed forward 포함)
        x = self.transformer_encoder(x)  # [B, 20, 1280]
        return x


class CustomTransformerEncoderLayer(nn.Module):
    """FC(Functional Connectivity) 행렬을 attention에 통합"""
    def __init__(self, fc_matrix_path, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.fc_matrix_path = fc_matrix_path
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, src_mask=None, is_causal=False, src_key_padding_mask=None):
        residual = x
        x = self.self_attn(x, self.fc_matrix_path)
        x = residual + self.dropout1(x)
        x = self.norm1(x)
        residual = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout2(x)
        x = self.norm2(x)
        return x


class CustomMultiheadAttention(nn.Module):
    """FC 행렬을 attention score에 추가"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, fc_matrix_path):
        B, T, E = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # FC matrix 추가
        fc_matrix = np.load(fc_matrix_path)
        fc_matrix = torch.from_numpy(fc_matrix).float().to(x.device)
        fc_matrix = fc_matrix.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T)
        attn_scores = attn_scores + fc_matrix * 0.7

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, E)
        return self.out_proj(attn_output)


# ============================================================================
# Low-Level Image Decoder (MindEye2 방식)
# ============================================================================

class LowLevelDecoder(nn.Module):
    """
    fMRI 임베딩에서 Low-level (blurry) 이미지 생성 (MindEye2 방식)

    input:
        fmri_emb: [B, 20, 1280]
    output:
        lowlevel_l1: [B, 4, 28, 28] - VAE latent space (L1 loss용, 224x224 이미지 기준)
        lowlevel_aux: [B, 49, 512] - ConvNext feature space (contrastive loss용)
    """
    def __init__(self, seq_len=20, embed_dim=1280):
        super().__init__()
        

        self.flatten_dim = seq_len * embed_dim  # 20 * 1280 = 25600

        # fmri_emb -> [B, 64, 7, 7] feature map (64*7*7 = 3136)
        self.blin1 = nn.Linear(self.flatten_dim, 64 * 7 * 7, bias=True)
        self.bdropout = nn.Dropout(0.3)
        self.bnorm = nn.GroupNorm(1, 64)

        # [B, 64, 7, 7] -> [B, 4, 28, 28] (VAE latent space, 3개 UpBlock으로 4x upsampling)
        self.bupsampler = Decoder(
            in_channels=64,
            out_channels=4,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[32, 64, 128],
            layers_per_block=1,
        )

        # [B, 64, 7, 7] -> [B, 49, 512] (ConvNext feature space)
        self.b_maps_projector = nn.Sequential(
            nn.Conv2d(64, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=True),
        )

    def forward(self, fmri_emb):
        """
        Input: fmri_emb [B, 20, 1280]
        Output:
            lowlevel_l1: [B, 4, 28, 28] - VAE latent (for L1 loss with vae.encode(image))
            lowlevel_aux: [B, 49, 512] - ConvNext features (for contrastive loss)
        """
        B = fmri_emb.size(0)
        x = fmri_emb.view(B, -1)  # [B, 25600]

        # linear -> dropout -> reshape -> groupnorm
        lowlevel = self.blin1(x)  # [B, 64*7*7]
        lowlevel = self.bdropout(lowlevel)
        lowlevel = lowlevel.reshape(B, 64, 7, 7).contiguous()  # [B, 64, 7, 7]
        lowlevel = self.bnorm(lowlevel)

        # L1 loss용: VAE latent space로 upsampling
        lowlevel_l1 = self.bupsampler(lowlevel)  # [B, 4, 28, 28]

        # ConvNext loss용: feature projection
        lowlevel_aux = self.b_maps_projector(lowlevel)  # [B, 512, 7, 7]
        lowlevel_aux = lowlevel_aux.flatten(2).permute(0, 2, 1)  # [B, 49, 512]

        return lowlevel_l1, lowlevel_aux


# ============================================================================
# Q-Former (그림의 (c) Q-former with fMRI)
# ============================================================================

class QFormerEncoderBlock(nn.Module):
    """Q-Former 블록: Self-Attention + Cross-Attention + FFN"""
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1, layer_norm_eps=1e-05):
        super().__init__()
        # Self-Attention
        self.self_ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.self_drop = nn.Dropout(dropout)
        # Cross-Attention
        self.cross_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_drop = nn.Dropout(dropout)
        # FFN (Feed forward)
        self.self_ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.ffn_drop = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, fmri_feats=None, attn_mask=None, n_q=None, do_cross=False):
        """
        Input: x [B, 257+257, 1280], fmri_feats [B, 20, 1280]
        Output: x [B, 257+257, 1280]
        """
        # Self-Attention
        residual = x
        x_norm = self.self_ln1(x)
        x_sa, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = residual + self.self_drop(x_sa)

        # Cross-Attention (query만 fMRI와 cross-attention)
        if do_cross and fmri_feats is not None:
            q = x[:, :n_q, :]
            q_res = q
            q_norm = self.cross_ln(q)
            q_ca, _ = self.cross_attn(q_norm, fmri_feats, fmri_feats, need_weights=False)
            q = q_res + self.cross_drop(q_ca)
            x = torch.cat([q, x[:, n_q:, :]], dim=1)

        # FFN (Feed forward)
        residual = x
        x_norm = self.self_ln2(x)
        x_ffn = self.fc2(self.ffn_drop(self.activation(self.fc1(x_norm))))
        x = residual + self.ffn_drop(x_ffn)
        return x


class QFormerEncoder(nn.Module):
    """
    Q-Former Encoder (CLIP ViT-H/14로 Self-Attention + FFN 초기화)

    그림 설명:
        (c) Q-former with fMRI
        - Self-attention: query+image 전체
        - Cross-attention: query만 fMRI와
        - Feed forward: 전체
    """
    def __init__(self, pretrained_model="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                 cross_attention_freq=2, num_query_tokens=257, dropout=0.1):
        super().__init__()
        vision = CLIPVisionModel.from_pretrained(pretrained_model)
        vcfg = getattr(vision.config, "vision_config", vision.config)

        self.hidden_size = vcfg.hidden_size  # 1280
        self.num_heads = vcfg.num_attention_heads  # 16
        self.intermediate_size = vcfg.intermediate_size  # 5120
        self.layer_norm_eps = vcfg.layer_norm_eps
        self.num_layers = vcfg.num_hidden_layers  # 32
        self.num_query_tokens = num_query_tokens
        self.cross_attention_freq = cross_attention_freq

        # 레이어별 cross-attn 적용 여부 미리 계산)(적용 레이어 (2, 4, 6, ... 번째))
        self._do_cross_map = [(cross_attention_freq > 0) and ((i + 1) % cross_attention_freq == 0)
                              for i in range(self.num_layers)]
        self.blocks = nn.ModuleList([
            QFormerEncoderBlock(self.hidden_size, self.num_heads, self.intermediate_size, dropout, self.layer_norm_eps)
            for _ in range(self.num_layers)
        ])

        # CLIP 가중치 로드 (Self-Attn + FFN만)
        for i, blk in enumerate(self.blocks):
            clip_blk = vision.vision_model.encoder.layers[i]
            blk.self_ln1.load_state_dict(clip_blk.layer_norm1.state_dict())
            blk.self_attn.load_state_dict(clip_blk.self_attn.state_dict())
            blk.self_ln2.load_state_dict(clip_blk.layer_norm2.state_dict())
            blk.fc1.load_state_dict(clip_blk.mlp.fc1.state_dict())
            blk.fc2.load_state_dict(clip_blk.mlp.fc2.state_dict())
        print(f"✅ Loaded CLIP pretrained weights for {self.num_layers} layers (Self-Attn + FFN)")

        del vision
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, x, fmri_feats, attn_mask=None):
        """
        Input: x [B, 257+257, 1280], fmri_feats [B, 20, 1280]
        Output: x [B, 257+257, 1280]
        """
        n_q = self.num_query_tokens
        for i, blk in enumerate(self.blocks):
            x = blk(x, fmri_feats=fmri_feats, attn_mask=attn_mask, n_q=n_q, do_cross=self._do_cross_map[i])
        return x


class FMRIImageAligner(nn.Module):
    """
    fMRI-Image Aligner: FIC (contrastive) + FIM (matching)

    그림 설명:
        Q-Former 출력 중 Query 부분만 사용 [B, 257, 1024]
        -> Pool (mean) + L2 norm -> [B, 1024]
        -> Stable Diffusion 2.1-unclip의 image_embeds로 사용

        FIM: CLS 뽑기 + logit -> [2B] (positive + negative matching)
    """
    def __init__(self, num_query_tokens=257, hidden_size=1280):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))

        # Q-Former Encoder
        self.q_former_encoder = QFormerEncoder(cross_attention_freq=2, num_query_tokens=num_query_tokens)

        # FIC: linear layer [1280 -> 1024]
        self.fic_classifier = nn.Linear(hidden_size, 1024)

        # FIM: CLS + logit linear layer [1280 -> 1]
        self.fim_classifier = nn.Linear(hidden_size, 1)

        # Losses
        self.fic_loss_fn = Blip2FICLoss()
        self.fim_loss_fn = Blip2FIMLoss()

    def build_mask(self, kind, n_q, n_t, device):
        """Attention mask: fic는 Q-T 상호 차단, fim은 전범위 허용"""
        L = n_q + n_t
        mask = torch.zeros(L, L, dtype=torch.bool, device=device)
        if kind == "fic":
            mask[:n_q, n_q:] = True  # Q -> T 차단
            mask[n_q:, :n_q] = True  # T -> Q 차단
        return mask

    def forward(self, fmri_emb, image_hidden_state, image_emb, device):
        """
        Training forward
        Input:
            fmri_emb [B, 20, 1280]
            image_hidden_state [B, 257, 1280]
            image_emb [B, 1024]
        Output:
            dict with fic_query_embedding, fic_image_embedding, fim_logits, fim_labels
        """
        B = fmri_emb.size(0)
        query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, 257, 1280]
        n_q = query_tokens.size(1)
        n_t = image_hidden_state.size(1)

        # === FIC Branch ===
        q_i_fic = torch.cat([query_tokens, image_hidden_state], dim=1)  # [B, 514, 1280]
        fic_attn_mask = self.build_mask("fic", n_q, n_t, device)
        q_i_hidden_fic = self.q_former_encoder(q_i_fic, fmri_emb, fic_attn_mask)  # [B, 514, 1280]

        # Query 부분만 사용 -> linear layer -> Pool + L2 norm
        q_tokens_fic = q_i_hidden_fic[:, :n_q, :]  # [B, 257, 1280]
        q_proj_fic = self.fic_classifier(q_tokens_fic)  # [B, 257, 1024]
        z_q_fic = q_proj_fic.mean(dim=1)  # Pool [B, 1024]
        z_q_fic = F.normalize(z_q_fic, dim=-1)  # L2 norm [B, 1024]

        # === FIM Branch ===
        perm = torch.randperm(B, device=device)
        query_tokens_2b = torch.cat([query_tokens, query_tokens], dim=0)
        image_feats_2b = torch.cat([image_hidden_state, image_hidden_state[perm]], dim=0)
        fmri_feats_2b = torch.cat([fmri_emb, fmri_emb], dim=0)
        fim_labels_2b = torch.cat([torch.ones(B, device=device), torch.zeros(B, device=device)])

        fim_attn_mask = self.build_mask("fim", n_q, n_t, device)
        q_i_fim = torch.cat([query_tokens_2b, image_feats_2b], dim=1)
        q_i_hidden_fim = self.q_former_encoder(q_i_fim, fmri_feats_2b, fim_attn_mask)  # [2B, 514, 1280]

        # Query CLS token (index 0) 사용
        query_cls = q_i_hidden_fim[:, 0, :]  # [2B, 1280]
        fim_logits = self.fim_classifier(query_cls).squeeze(-1)  # [2B]

        return {
            "fic_query_embedding": z_q_fic,      # [B, 1024]
            "fic_image_embedding": image_emb,   # [B, 1024]
            "fim_logits": fim_logits,           # [2B]
            "fim_labels": fim_labels_2b,        # [2B]
        }

    def compute_loss(self, reps):
        loss_fic = self.fic_loss_fn(reps["fic_query_embedding"], reps["fic_image_embedding"])
        loss_fim = self.fim_loss_fn(reps["fim_logits"], reps["fim_labels"])
        return {"loss_fic": loss_fic, "loss_fim": loss_fim}

    def inference(self, fmri_emb):
        """
        Inference (이미지 없이 fMRI만으로)
        Input: fmri_emb [B, 20, 1280]
        Output: [B, 1024] - Stable UnCLIP의 image_embeds로 사용
        """
        B = fmri_emb.size(0)
        query_tokens = self.query_tokens.expand(B, -1, -1)
        query_hidden = self.q_former_encoder(query_tokens, fmri_emb, attn_mask=None)
        q_proj = self.fic_classifier(query_hidden)
        z_q = q_proj.mean(dim=1)
        z_q = F.normalize(z_q, dim=-1)
        return z_q


# ============================================================================
# Loss Functions
# ============================================================================

class Blip2FICLoss(nn.Module):
    """FIC Loss: InfoNCE (contrastive learning)"""
    def __init__(self, temperature_init=0.07, learnable_temperature=True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature_init)), requires_grad=learnable_temperature)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, query_emb, image_emb):
        B = query_emb.size(0)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_q2i = scale * (query_emb @ image_emb.t())
        logits_i2q = scale * (image_emb @ query_emb.t())
        target = torch.arange(B, device=query_emb.device)
        return 0.5 * (self.ce(logits_q2i, target) + self.ce(logits_i2q, target))


class Blip2FIMLoss(nn.Module):
    """FIM Loss: BCE (matching)"""
    def forward(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).float())


# ============================================================================
# Complete Model
# ============================================================================

class ConnecToMind2(nn.Module):
    """
    ConnecToMind2 - Complete Model

    Loss = fMRI-image contrastive learning (FIC)
         + fMRI-image matching (FIM)
         + low-level L1 loss (VAE latent space)
         + low-level ConvNext contrastive loss

    output (training):
        fmri_emb: [B, 20, 1280] - Connectome-Transformer 출력
        lowlevel_l1: [B, 4, 28, 28] - VAE latent (for L1 loss, 224x224 이미지 기준)
        lowlevel_aux: [B, 49, 512] - ConvNext features (for contrastive loss)
        clip_embedding: [B, 1024] - Q-Former 출력 (Pool + L2 norm)
        loss_fic, loss_fim: Q-Former losses

    Stable Diffusion 2.1-unclip inputs:
        - image_embeds: Q-Former 출력 [B, 1024] (Pool + L2 norm)
        - image: VAE decode(lowlevel_l1) for image condition
    """
    def __init__(self, seq_len=20, input_dim=2056, embed_dim=1280, num_transformer_layers=1,
                 is_fc=False, fc_matrix_path=""):
        super().__init__()

        # 1. CLIP Image Encoder (frozen)
        self.clip_encoder = CLIPImageEncoder(freeze=True)

        # 2. Connectome-Transformer
        self.connectome_transformer = ConnectomeTransformer(seq_len=seq_len, input_dim=input_dim, embed_dim=embed_dim, num_layers=num_transformer_layers, is_fc=is_fc, fc_matrix_path=fc_matrix_path)

        # 3. Low-Level Decoder (MindEye2 방식)
        self.low_level_decoder = LowLevelDecoder(seq_len=seq_len, embed_dim=embed_dim)

        # 4. Q-Former (FMRIImageAligner)
        self.q_former = FMRIImageAligner(num_query_tokens=257, hidden_size=1280)

    def forward(self, fmri, images, device):
        """
        Training forward

        Input:
            fmri [B, 20, 2056]
            images [B, 3, 224, 224]
            device: torch device

        Output: dict
            fmri_emb: [B, 20, 1280]
            lowlevel_l1: [B, 4, 28, 28] - for L1 loss with vae.encode(image)
            lowlevel_aux: [B, 49, 512] - for soft_cont_loss with ConvNext
            clip_embedding: [B, 1024]
            loss_fic: scalar
            loss_fim: scalar
        """
        # fMRI path
        fmri_emb = self.connectome_transformer(fmri)  # [B, 20, 1280]
        lowlevel_l1, lowlevel_aux = self.low_level_decoder(fmri_emb)  # [B, 4, 28, 28], [B, 49, 512]

        # Image path
        image_hidden, image_emb = self.clip_encoder(images)  # [B, 257, 1280], [B, 1024]

        # Q-Former
        reps = self.q_former(fmri_emb, image_hidden, image_emb, device)
        losses = self.q_former.compute_loss(reps)

        return {
            "fmri_emb": fmri_emb,
            "lowlevel_l1": lowlevel_l1,      # for L1 loss
            "lowlevel_aux": lowlevel_aux,    # for soft_cont_loss
            "clip_embedding": reps["fic_query_embedding"],
            "loss_fic": losses["loss_fic"],
            "loss_fim": losses["loss_fim"],
        }

    def inference(self, fmri):
        """
        Inference (이미지 없이)

        Input: fmri [B, 20, 2056]
        Output:
            clip_embedding: [B, 1024] - Stable UnCLIP의 image_embeds로 사용
            lowlevel_l1: [B, 4, 28, 28] - VAE decode하면 blurry image
        """
        fmri_emb = self.connectome_transformer(fmri)
        lowlevel_l1, _ = self.low_level_decoder(fmri_emb)
        clip_embedding = self.q_former.inference(fmri_emb)
        return {"clip_embedding": clip_embedding, "lowlevel_l1": lowlevel_l1}


# ============================================================================
# Model Factory
# ============================================================================

def get_model(args):
    """
    모델 생성

    args 필요 속성:
        - seq_len, input_dim, embed_dim, num_transformer_layers
        - is_fc, fc_matrix_path
        - cache_dir: pretrained model cache 경로

    returns:
        connectomind2: 메인 모델
        unclip_pipeline: Stable UnCLIP 2.1 pipeline
        vae: VAE encoder (for L1 loss)
        cnx: ConvNext XL (for contrastive loss)
        l1: L1 loss function
    """


    # 메인 모델
    connectomind2 = ConnecToMind2(
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        num_transformer_layers=args.num_transformer_layers,
        is_fc=args.is_fc,
        fc_matrix_path=args.fc_matrix_path,
    )

    # high-level reconstruction 용도 -> Stable UnCLIP 2.1 pipeline
    print("Loading Stable UnCLIP 2.1 pipeline...")
    unclip_pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16)
    print("✅ Stable UnCLIP 2.1 loaded")

    # low-level reconstruction 용도 -> VAE (for L1 loss - low-level)
    print("Loading VAE...")
    try:
        sd_model_dir = os.path.join(args.cache_dir, "models--lambdalabs--sd-image-variations-diffusers", "snapshots")
        snapshot_name = os.listdir(sd_model_dir)[0]
        snapshot_path = os.path.join(sd_model_dir, snapshot_name)
        sd_pipe = DiffusionPipeline.from_pretrained(snapshot_path)
    except Exception as e:
        print(f"[!] 로컬 snapshot 로딩 실패, 온라인에서 로드: {e}")
        sd_pipe = DiffusionPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", cache_dir=args.cache_dir)
    vae = sd_pipe.vae
    vae.eval().requires_grad_(False)
    print("✅ VAE loaded")

    # ConvNext XL (for contrastive loss - low-level)
    print("Loading ConvNext XL...")
    cnx_path = os.path.join(args.cache_dir, "convnext_xlarge_alpha0.75_fullckpt.pth")
    cnx = ConvnextXL(cnx_path)
    cnx.requires_grad_(False)
    cnx.eval()
    print("✅ ConvNext XL loaded")

    # L1 loss
    l1 = nn.L1Loss()

    return {
        "connectomind2": connectomind2,
        "unclip_pipeline": unclip_pipeline,
        "vae": vae,
        "cnx": cnx,
        "l1": l1,
    }
