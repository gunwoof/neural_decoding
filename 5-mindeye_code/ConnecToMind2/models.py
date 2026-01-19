"""
ConnecToMind2 - Model2 Implementation (Based on New Architecture Diagram)

Architecture (from diagram):
    fMRI [B, 200, (roi+padding)]
        -> (a) Region-level embedding -> [B, 200, 768]
        -> (b) Connectome-Q-former -> [B, 201, 768]
        -> Linear layer -> [B, 257, 768]
        -> L2 norm -> [B, 257, 768]
        -> Versatile Diffusion -> Reconstructed Image [B, 512, 512]

    Image [B, 224, 224]
        -> CLIP ViT-L/14 -> Last hidden [B, 257, 1024]
        -> Linear layer + L2 norm -> [B, 257, 768]

Loss = FIR Loss (fMRI embedding vs CLIP embedding, MSE)
     + Cross Entropy Loss (CLS token)
     + Low-level Loss (L1 with target image)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from transformers import CLIPVisionModel
from diffusers import DiffusionPipeline
from diffusers.models.autoencoder_kl import Decoder

from convnext import ConvnextXL


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
        self.clip_model = CLIPVisionModel.from_pretrained(pretrained_model)

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # CLIP ViT-L/14 hidden size: 1024
        self.clip_hidden_size = 1024
        self.output_dim = 768

        # Linear layer: [B, 257, 1024] -> [B, 257, 768]
        self.proj = nn.Linear(self.clip_hidden_size, self.output_dim)

    def forward(self, images):
        """
        Input: images [B, 3, 224, 224]
        Output:
            hidden_state [B, 257, 768] - Linear + L2 norm (for MSE Loss)
            cls_token [B, 768] - CLS token (for Cross Entropy Loss)
        """
        outputs = self.clip_model(images, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state  # [B, 257, 1024]

        # Linear layer + L2 norm
        hidden_state = self.proj(last_hidden)  # [B, 257, 768]
        hidden_state = F.normalize(hidden_state, dim=-1)  # L2 norm

        return hidden_state


# ============================================================================
# Region-level Embedding (그림의 (a))
# ============================================================================

class RegionLevelEmbedding(nn.Module):
    """
    (a) Region-level embedding

    그림 설명:
        task-fMRI [B, 200, (roi+padding)]
        -> Flatten -> Linear projection -> [B, 200, 768]
    """
    def __init__(self, seq_len=200, input_dim=3291, embed_dim=768):
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
        Input: x [B, 200, input_dim]
        Output: x [B, 200, 768]
        """
        # Region-level embedding (각 ROI별 linear)
        x = torch.einsum("btd,tdh->bth", x, self.linear_weight)  # [B, 200, 768]
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

    def forward(self, x, fmri_feats, attn_mask=None, do_cross=True, n_q=None):
        """
        Input:
            x [B, L, 768] - query tokens (+ optional CLIP hidden states)
                           L = n_q (inference) or n_q + n_t (training with mask)
            fmri_feats [B, 200, 768] - fMRI embeddings
            attn_mask: attention mask for Q-T separation
            do_cross: whether to do cross-attention in this layer
            n_q: number of query tokens (for extracting query part in cross-attention)
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
            q_ca, _ = self.cross_attn(q_norm, fmri_feats, fmri_feats, need_weights=False)
            q = q_res + self.cross_drop(q_ca)
            x = torch.cat([q, x[:, n_q:, :]], dim=1)  # Query + 나머지 다시 concat

        # Feed Forward
        residual = x
        x_norm = self.ffn_ln(x)
        x_ffn = self.fc2(self.ffn_drop(self.activation(self.fc1(x_norm))))
        x = residual + self.ffn_drop(x_ffn)

        return x


class ConnectomeQFormer(nn.Module):
    """
    (b) Connectome-Q-Former (initialized from CLIP ViT-L/14)

    그림 설명:
        [B, 200, 768] (fMRI) + query tokens + CLIP hidden states
        -> Connectome-Q-Former blocks (with attention mask)
        -> [B, 201, 768]

    Query tokens: 201개 (200 ROI + 1 CLS token)
    Weights initialized from openai/clip-vit-base-patch16
    Cross-attention: 2 layer마다 한 번 (BLIP-2 방식)
    Layers: 12 (CLIP ViT-B/16 기준)

    마스크 기반 Self-attention (models.py 방식):
        Query + CLIP hidden states를 concat하여 처리
        마스크로 Q-T 상호 attend 차단 -> 독립적 처리
    """
    def __init__(self, hidden_size=768, num_heads=12, num_layers=12,
                 num_query_tokens=201, dropout=0.1, cross_attention_freq=2,
                 clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.cross_attention_freq = cross_attention_freq

        # Cross-attention 적용 여부 미리 계산 (0, 2, 4, ... 번째 layer) - BLIP-2 방식
        self._do_cross_map = [(cross_attention_freq > 0) and (i % cross_attention_freq == 0)
                              for i in range(num_layers)]

        # Learnable query tokens [1, 201, 768] - position embedding 없음
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        # Connectome-Q-Former blocks (12 layers, CLIP ViT-B/16 기준)
        self.blocks = nn.ModuleList([
            ConnectomeQFormerBlock(hidden_size, num_heads, hidden_size * 4, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.final_ln = nn.LayerNorm(hidden_size)

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

    def forward(self, fmri_emb, clip_hidden=None, use_mask=True):
        """
        Input:
            fmri_emb [B, 200, 768]
            clip_hidden [B, 257, 768] (optional) - CLIP hidden states for masked self-attention
            use_mask: True=Q-T 상호 차단 (FIR용), False=전범위 허용 (FIM용)
        Output:
            query_output [B, 201, 768]
        """
        B = fmri_emb.size(0)
        device = fmri_emb.device

        # fMRI features (position embedding 없음)
        fmri_feats = fmri_emb  # [B, 200, 768]

        # Expand query tokens for batch (position embedding 없음)
        query = self.query_tokens.expand(B, -1, -1)  # [B, 201, 768]

        # Concat query + CLIP hidden states (if provided)
        if clip_hidden is not None:
            # [B, 201, 768] + [B, 257, 768] -> [B, 458, 768]
            x = torch.cat([query, clip_hidden], dim=1)
            n_q = self.num_query_tokens
            n_t = clip_hidden.size(1)
            # use_mask=False: 전범위 허용 (FIM용) vs use_mask=True: Q-T 상호 차단 (MSE용)
            attn_mask = self.build_mask(n_q, n_t, device) if use_mask else None
        else:
            x = query
            attn_mask = None

        # Pass through Connectome-Q-Former blocks
        for i, block in enumerate(self.blocks):
            x = block(x, fmri_feats, attn_mask=attn_mask, do_cross=self._do_cross_map[i], n_q=self.num_query_tokens)

        # Extract query part only (if CLIP hidden was concatenated)
        if clip_hidden is not None:
            query_output = x[:, :self.num_query_tokens, :]  # [B, 201, 768]
        else:
            query_output = x

        # Final LayerNorm
        query_output = self.final_ln(query_output)  # [B, 201, 768]

        return query_output


# ============================================================================
# Low-Level Image Decoder
# ============================================================================

class LowLevelDecoder(nn.Module):
    """
    fMRI 임베딩에서 Low-level (blurry) 이미지 생성 (MindEye2 방식)

    input:
        fmri_emb: [B, 200, 768]
    output:
        lowlevel_l1: [B, 4, 28, 28] - VAE latent space (L1 loss용, 224x224 이미지 기준)
        lowlevel_aux: [B, 49, 512] - ConvNext feature space (contrastive loss용)
    """
    def __init__(self, seq_len=200, embed_dim=768):
        super().__init__()

        self.flatten_dim = seq_len * embed_dim  # 200 * 768 = 153600

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
        Input: fmri_emb [B, 200, 768]
        Output:
            lowlevel_l1: [B, 4, 28, 28] - VAE latent (for L1 loss with vae.encode(image))
            lowlevel_aux: [B, 49, 512] - ConvNext features (for contrastive loss)
        """
        B = fmri_emb.size(0)
        x = fmri_emb.view(B, -1)  # [B, 153600]

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
# Output Projection (Linear layer + L2 norm)
# ============================================================================

class OutputProjection(nn.Module):
    """
    Q-Former 출력을 CLIP space로 projection (Transpose 방식)

    그림 설명:
        Q-Former output [B, 201, 768]
        -> Transpose -> [B, 768, 201]
        -> Linear(201, 257) -> [B, 768, 257]
        -> Transpose -> [B, 257, 768]
        -> L2 norm -> [B, 257, 768]

    파라미터 수: 201 * 257 = 51,657 (Flatten 방식 대비 ~600,000배 적음)
    """
    def __init__(self, input_tokens=201, output_tokens=257, hidden_size=768):
        super().__init__()
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.hidden_size = hidden_size

        # [B, 768, 201] -> [B, 768, 257] (각 dimension별 독립 projection)
        self.proj = nn.Linear(input_tokens, output_tokens)

    def forward(self, x):
        """
        Input: x [B, 201, 768]
        Output: x [B, 257, 768] (L2 normalized)
        """
        x = x.transpose(1, 2)  # [B, 768, 201]
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


class CrossEntropyLoss(nn.Module):
    """FIM Loss: BCE (matching)"""
    def forward(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).float())


# ============================================================================
# Complete Model
# ============================================================================

class ConnecToMind2(nn.Module):
    """
    ConnecToMind2 - Model2 (New Architecture)

    Architecture:
        fMRI [B, 200, input_dim]
            -> (a) Region-level embedding -> [B, 200, 768]
            -> (b) Connectome-Q-Former -> [B, 201, 768]
            -> Linear layer -> [B, 257, 768]
            -> L2 norm -> [B, 257, 768]

        Image [B, 3, 224, 224]
            -> CLIP ViT-L/14 -> [B, 257, 1024]
            -> Linear layer + L2 norm -> [B, 257, 768]

    Loss = FIR Loss (fMRI embedding vs CLIP embedding, MSE)
         + Cross Entropy Loss (CLS token contrastive)
         + Low-level Loss (VAE L1 + ConvNext Contrastive)

    Output (training):
        fmri_proj: [B, 257, 768] - Q-Former output (Linear + L2 norm)
        clip_proj: [B, 257, 768] - CLIP output (Linear + L2 norm)
        fmri_cls: [B, 768] - fMRI CLS token
        clip_cls: [B, 768] - CLIP CLS token
        lowlevel_l1: [B, 4, 28, 28] - VAE latent (for L1 loss)
        lowlevel_aux: [B, 49, 512] - ConvNext features (for contrastive loss)
        loss_fir: FIR loss
        loss_cls: Cross entropy loss

    Versatile Diffusion inputs:
        - image_embeds: fmri_proj [B, 257, 768]
        - image: VAE decode(lowlevel_l1) for image condition
    """
    def __init__(self, seq_len=200, input_dim=2056, embed_dim=768,
                 num_qformer_layers=12, num_query_tokens=201):
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

        # 3. Connectome-Q-Former (initialized from CLIP)
        self.connectome_qformer = ConnectomeQFormer(
            hidden_size=embed_dim,
            num_heads=12,
            num_layers=num_qformer_layers,
            num_query_tokens=num_query_tokens,
            dropout=0.1
        )

        # 4. Output Projection: [B, 201, 768] -> [B, 257, 768]
        self.output_proj = OutputProjection(
            input_tokens=num_query_tokens,
            output_tokens=257,
            hidden_size=embed_dim
        )

        # 5. Low-Level Decoder
        self.low_level_decoder = LowLevelDecoder(seq_len=seq_len, embed_dim=embed_dim)

        # 6. FIM classifier: CLS token [768] -> logit [1]
        self.fim_classifier = nn.Linear(embed_dim, 1)

        # 7. Loss functions
        self.fir_loss_fn = FIRLoss()
        self.fim_loss_fn = CrossEntropyLoss()  # BCE loss

    def forward(self, fmri, images, device):
        """
        Training forward (Q-Former 2번 호출: FIR용 마스크 O, FIM용 마스크 X)

        FIR: Q-T 상호 차단 마스크 사용 (Query가 CLIP을 직접 보면 cheating)
        FIM: 마스크 없음 (Query가 CLIP을 보고 matching 판단)

        Input:
            fmri [B, 200, input_dim]
            images [B, 3, 224, 224]
            device: torch device

        Output: dict
            fmri_proj: [B, 257, 768] - for FIR loss
            clip_proj: [B, 257, 768] - for FIR loss
            lowlevel_l1: [B, 4, 28, 28] - for L1 loss with vae.encode(image)
            lowlevel_aux: [B, 49, 512] - for soft_cont_loss with ConvNext
            loss_fir: scalar
            loss_fim: scalar
        """
        B = fmri.size(0)

        # === Image path (CLIP) ===
        clip_proj = self.clip_encoder(images)  # [B, 257, 768]

        # === fMRI path ===
        # (a) Region-level embedding
        fmri_emb = self.region_embedding(fmri)  # [B, 200, 768]

        # === FIR Branch (마스크 O: Q-T 상호 차단) ===
        qformer_out_fir = self.connectome_qformer(fmri_emb, clip_proj, use_mask=True)  # [B, 201, 768]
        fmri_proj = self.output_proj(qformer_out_fir)  # [B, 257, 768]

        # === FIM Branch (마스크 X: 전범위 허용) ===
        perm = torch.randperm(B, device=device)
        fmri_emb_2b = torch.cat([fmri_emb, fmri_emb], dim=0)  # [2B, 200, 768]
        clip_proj_2b = torch.cat([clip_proj, clip_proj[perm]], dim=0)  # [2B, 257, 768]
        fim_labels = torch.cat([torch.ones(B, device=device), torch.zeros(B, device=device)])  # [2B]

        qformer_out_fim = self.connectome_qformer(fmri_emb_2b, clip_proj_2b, use_mask=False)  # [2B, 201, 768]

        # Low-level decoder (원본 fmri_emb만 사용)
        lowlevel_l1, lowlevel_aux = self.low_level_decoder(fmri_emb)  # [B, 4, 28, 28], [B, 49, 512]

        # === Compute FIR loss ===
        loss_fir = self.fir_loss_fn(fmri_proj, clip_proj)

        # === FIM Loss ===
        query_cls_2b = qformer_out_fim[:, 0, :]  # [2B, 768]
        fim_logits = self.fim_classifier(query_cls_2b).squeeze(-1)  # [2B]
        loss_fim = self.fim_loss_fn(fim_logits, fim_labels)

        return {
            "fmri_proj": fmri_proj,          # [B, 257, 768]
            "clip_proj": clip_proj,          # [B, 257, 768]
            "lowlevel_l1": lowlevel_l1,      # [B, 4, 28, 28]
            "lowlevel_aux": lowlevel_aux,    # [B, 49, 512]
            "loss_fir": loss_fir,
            "loss_fim": loss_fim,
        }

    def inference(self, fmri):
        """
        Inference (이미지 없이)

        Input: fmri [B, 200, input_dim]
        Output:
            fmri_proj: [B, 257, 768] - Versatile Diffusion의 image_embeds로 사용
            lowlevel_l1: [B, 4, 28, 28] - VAE decode하면 blurry image
        """
        # (a) Region-level embedding
        fmri_emb = self.region_embedding(fmri)  # [B, 200, 768]

        # (b) Connectome-Q-Former
        qformer_out = self.connectome_qformer(fmri_emb)  # [B, 257, 768]

        # Linear layer + L2 norm
        fmri_proj = self.output_proj(qformer_out)  # [B, 257, 768]

        # Low-level decoder
        lowlevel_l1, _ = self.low_level_decoder(fmri_emb)  # [B, 4, 28, 28]

        return {
            "fmri_proj": fmri_proj,      # [B, 257, 768]
            "lowlevel_l1": lowlevel_l1,  # [B, 4, 28, 28]
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
        cnx: ConvNext XL (for contrastive loss)
        l1: L1 loss function
    """
    # 캐시 경로 (로컬 우선, 없으면 온라인에서 다운로드)
    cache_dir = args.cache_dir

    # 메인 모델
    connectomind2 = ConnecToMind2(
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        num_qformer_layers=args.num_qformer_layers,
        num_query_tokens=args.num_query_tokens,
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

    # ConvNext XL (for contrastive loss - low-level)
    print("Loading ConvNext XL...")
    cnx_path = os.path.join(cache_dir, "convnext_xlarge_alpha0.75_fullckpt.pth")
    cnx = ConvnextXL(cnx_path)
    cnx.requires_grad_(False)
    cnx.eval()
    print("✅ ConvNext XL loaded")

    # L1 loss
    l1 = nn.L1Loss()

    return {
        "connectomind2": connectomind2,
        "versatile_diffusion": versatile_diffusion,
        "vae": vae,
        "cnx": cnx,
        "l1": l1,
    }
