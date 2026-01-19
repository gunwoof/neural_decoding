import os
import math
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import PIL
from functools import partial
import glob
from scipy.stats import pearsonr
import gc

# q-former에 CLIP ViT-H/14의 self-attention + ffn를 init
from transformers import CLIPVisionModel
from transformers import CLIPVisionModelWithProjection

# stable unclip 2.1
from diffusers import StableUnCLIPImg2ImgPipeline

# 학습하는 모델을 담을 class
class ConnecToMind2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class ConnectomeTransformer(nn.Module):
    def __init__(self, seq_len=20, input_dim=2056, embed_dim=1280, nhead=8, num_layers=8, is_position=False, is_fc=False, fc_matrix_path=""):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.is_position = is_position

        # roi별로 다른 weight의 linear layer (seq_len x input_dim x embed_dim) -> einsum 사용
        self.linear1_weight = nn.Parameter(torch.empty(seq_len, input_dim, embed_dim))
        for t in range(seq_len):
            init.xavier_uniform_(self.linear1_weight[t]) # xavier_uniform 초기화
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)

        # positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        if is_fc:
            encoder_layer = CustomTransformerEncoderLayer(fc_matrix_path=fc_matrix_path, d_model=embed_dim, nhead=nhead)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # num_layers개 쌓음
            
    def forward(self, x):
        '''
            x.shape [batch_size, roi개수, (voxel개수+padding)] -> [batch_size, roi개수, 1280]
        '''
        #### region-level fMRI embeddings ####
        x = torch.einsum("btd,tdh->bth", x, self.linear1_weight) # [B, roi개수, (voxel개수+padding)] -> [B, roi개수, 1280] - 각 roi마다 linear layer
        x = self.layernorm1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        #### Connectome-Transformer ####
        x = x + self.pos_embedding # positional embedding
        x = self.transformer_encoder(x)  # [B, 20, 1280] -> [B, 20, 1280]

        return x
    
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, fc_matrix_path, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.fc_matrix_path = fc_matrix_path
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        self.activation = F.gelu

    def forward(self, x, src_mask=None, is_causal=False, src_key_padding_mask=None):
        # Self-attention block
        residual = x
        x = self.self_attn(x, self.fc_matrix_path)
        x = residual + self.dropout1(x)
        x = self.norm1(x)

        # Feedforward block
        residual = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout2(x)
        x = self.norm2(x)

        return x

class CustomMultiheadAttention(nn.Module):
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
        
        # q, k, v 한 번에 계산하고 쪼갬
        qkv = self.qkv_proj(x)  # (B, T, 3E)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # FC 사용
        fc_matrix = np.load(fc_matrix_path)        # shape (T, T)
        fc_matrix = torch.from_numpy(fc_matrix).float().to(x.device)
        fc_matrix = fc_matrix.unsqueeze(0).unsqueeze(0)
        fc_matrix = fc_matrix.expand(B, 1, T, T)
        attn_scores = attn_scores + fc_matrix * 0.7

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, E)

        out = self.out_proj(attn_output)

        return out

class FMRIImageAligner(nn.Module):
    """
    fMRI ↔ Image 정렬 (FIC) + 멀티모달 매칭 (FIM) 모델

    input:
        fmri_emb: [B, 20, 1280]   (fMRI embedding)
        image_hidden_state: [B, 257, 1280]  (ViT-H/14 embedding; idx0 = CLS/global)  
        image_emb: [B, 1024]  (CLIP image embedding; for FIC loss)

    내부 동작:
      FIC branch:
        query+image [B,257+257,1280], mask [257+257,257+257] -> q-former -> fic_query_embedding [B,257,1280], fic_image_embedding [B,257,1280]
      FIM branch:
        query+image [2B,257+257,1280], mask [257+257,257+257] -> q-former -> fim_logits [2B]

    output:
        {
            "fic_query_embedding" [B,257,1280]
            "fic_image_embedding" [B,257,1280]
            "fim_logits" [2B]
            "fim_labels" [2B]
        }
    """
    def __init__(
        self,
        num_query_tokens=257,     # ViT-H/14 default
        hidden_size=1280,         # ViT-H/14 default
        cross_attention_freq=2,  # BLIP-2 default
        dropout=0.1,             # BLIP-2 default
        layer_norm_eps=1e-12,    # BLIP-2 default
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens

        # #### projection ####
        # self.fmri_proj = nn.Linear(1280, hidden_size)   # [B,20,1280] -> [B,20,1280]
        # self.image_proj = nn.Linear(1280, hidden_size)   # [B,257,1280] -> [B,257,1280]

        #### learnable query tokens (shared across FIC / FIM) ####
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size)) # [1,257,1280] -> broadcast to [B,257,1280]

        #### q_former ####
        self.q_former = QFormerEncoder(
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            cross_attention_freq=cross_attention_freq,
            num_query_tokens=num_query_tokens,
        )

        #### heads ####
        # FIC classifier head
        self.fic_classifier = nn.Linear(hidden_size, 1024)
        # FIM classifier head
        self.fim_classifier = nn.Linear(hidden_size, 1)

        #### losses ####
        # FIC: align loss
        self.fic_loss_fn = Blip2FICLoss()
        # FIM: positive match + negative match
        self.fim_loss_fn = Blip2FIMLoss()
    
    def build_mask(self, kind, n_q, n_t, device):
        """
            mask: [L, L] bool 행렬 (True=차단, False=허용)
            if fim : 전범위 양방향
            if fic : Q와 T 상호 차단(블록 대각)
        """
        # nn.MultiheadAttention은 True:무시 & False:실행 -> zero로 초기화 = 모두 허용
        L = n_q + n_t
        mask = torch.zeros(L, L, dtype=torch.bool, device=device)

        if kind == "fim":
            return mask

        if kind == "fic":
            mask[:n_q, n_q:] = True                        # Q -> T 차단
            mask[n_q:, :n_q] = True                         # T -> Q 차단
            return mask

    def forward(self, args, fmri_emb, image_hidden_state, image_emb):
        """
            fmri_emb: [B, 20, 1280]   (fMRI embedding)
            image_hidden_state: [B, 257, 1280]  (ViT-H/14 embedding; idx0 = CLS/global)  
            image_emb: [B, 1024]  (CLIP image embedding; for FIC loss)
            returns: dict
                {   
                    "fic_query_embedding" [B,1024]
                    "fic_image_embedding" [B,1024]
                    "fim_logits" [2B]
                    "fim_labels" [2B]
                }
        """
        B = fmri_emb.size(0)
        # query tokens
        query_tokens = self.query_tokens.expand(B, -1, -1)  # [B,257,1280]

        n_q = query_tokens.size(1)   # 보통 257
        n_t = image_hidden_state.size(1)    # 보통 257

        #### FIC branch ####
        q_i_fic = torch.cat([query_tokens, image_hidden_state], dim=1) # [B,257+257,1280]
        # q-former에 넣을 mask들 생성
        fic_attn_mask = self.build_mask("fic", n_q, n_t, device=args.device)  # [257+257,257+257] bool
        # q-former(self attention with all + cross-attn between only query and fMRI + FFN with all)
        q_i_hidden_fic = self.q_former(q_i_fic, fmri_emb, fic_attn_mask)  # [B,257+257,1280]
        # query projection + mean pooling + l2norm
        q_tokens_fic = q_i_hidden_fic[:, :n_q, :] #  query 토큰만 선택 [B,257,1280]
        q_proj_fic = self.fic_classifier(q_tokens_fic) # projection [B,257,1024]
        z_q_fic = q_proj_fic.mean(dim=1) # mean pooling [B,1024]
        z_q_fic = F.normalize(z_q_fic, dim=-1) # l2norm [B,1024]

        #### FIM branch ####
        # negative를 위해 image embedding 섞기
        perm = torch.randperm(B, device=args.device)
        # 2B(positive+negative)로 확장
        query_tokens_2b = torch.cat([query_tokens, query_tokens], dim=0) # [2B,257,1280]
        image_feats_2b = torch.cat([image_hidden_state, image_hidden_state[perm]], dim=0) # [2B,257,1280]
        fmri_feats_2b  = torch.cat([fmri_emb, fmri_emb], dim=0) # [2B,20,1280]
        fim_labels_2b = torch.cat([torch.ones(B,device=args.device), torch.zeros(B,device=args.device)], dim=0) # [2B]
        # q-former에 넣을 mask들 생성
        fim_attn_mask = self.build_mask("fim", n_q, n_t, device=args.device)  # [257+257,257+257] bool        
        # q-former(self attention with all + cross-attn between only query and fMRI + FFN with all)
        q_i_fim = torch.cat([query_tokens_2b, image_feats_2b], dim=1) # [2B,257+257,1280]
        q_i_hidden_fim = self.q_former(q_i_fim, fmri_feats_2b, fim_attn_mask)  # [2B,514,1280]
        # cls + logit linear layer(1280,1)
        fused_cls = q_i_hidden_fim[:, self.num_query_tokens, :]  # image부분의 cls token 뽑기 [2B,1280]
        fim_logits = self.fim_classifier(fused_cls).squeeze(-1)  # logit계산(linear layer) [2B,1280] -> [2B]

        reps = {
            "fic_query_embedding": z_q_fic,  # [B,1024]
            "fic_image_embedding": image_emb,  # clip에서 image_embeds를 바로 가져옴 [B,1024]
            "fim_logits": fim_logits,     # [2B]
            "fim_labels": fim_labels_2b,  # [2B]
        }
        return reps

    # compute_loss
    def compute_loss(self, representations):
        """
        representations: dict output from forward()
            must contain q_hidden, t_hidden, itm_logits
        itm_labels: [B] or [B,1] with {0,1}, optional
        """
        fic_query_embedding = representations["fic_query_embedding"]     # [B,1024]
        fic_image_embedding = representations["fic_image_embedding"]     # [B,1024]
        fim_logits = representations["fim_logits"] # [2B]
        fim_labels = representations["fim_labels"] # [2B]

        #### FIC MSE (fine-grained alignment of all tokens) ####
        loss_fic = self.fic_loss_fn(fic_query_embedding, fic_image_embedding) # scalar

        #### FIM BCE (binary match) ####
        loss_fim = self.fim_loss_fn(fim_logits, fim_labels)  # scalar

        return {
            "loss_fic": loss_fic, # scalar
            "loss_fim": loss_fim,  # scalar 
        }
    
    def inference(self, fmri_emb):
        """
            input:
                fmri_emb: [B, 20, 1280]   (fMRI embedding)
            returns: 
                fic_query_embedding: [B,257,1280]
        """
        B = fmri_emb.size(0)
        # query tokens
        query_tokens = self.query_tokens.expand(B, -1, -1)  # [B,257,1280]
        query_hidden = self.q_former(query_tokens, fmri_emb, attn_mask=None)  # [B, 257, 1280]
        # query projection + mean pooling + l2norm
        q_proj_fic = self.fic_classifier(query_hidden) # projection [B,257,1024]
        z_q_fic = q_proj_fic.mean(dim=1) # mean pooling [B,1024]
        z_q_fic = F.normalize(z_q_fic, dim=-1) # l2norm [B,1024]
        return z_q_fic


class QFormerEncoder(nn.Module):
    """
        - QFormerEncoderBlock을 CLIPVisionModel(laion/CLIP-ViT-H-14-laion2B-s32B-b79K)기준으로 init 
        - QFormerEncoderBlock을 num_layers개 쌓음
        - cross_attention_freq 간격으로 cross-attn 블록 배치 (예: 2 => 2,4,6,...번째 레이어에서 cross-attn)
        - 입력: [B, 257+257, 1280]  (concat of [queries, images]), fmri_feats: [B, 20, 1280], attn_mask:  [257+257, 257+257] (train) or None (inference)
        - 출력: [B, 257+257, 1280]
    """
    def __init__(
        self,
        pretrained_vision_model = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        cross_attention_freq = 2,
        num_query_tokens = 257,
        dropout = 0.1, 
    ):
        super().__init__()
        vision = CLIPVisionModel.from_pretrained(pretrained_vision_model)
        # getattr로 vision.config 접근 (vision.config.vision_config가 있으면 사용, 없으면 vision.config 사용) -> 버전 차이 대응
        vcfg = getattr(vision.config, "vision_config", vision.config)

        # CLIP-ViT-H-14을 기준으로 q-former setting 
        self.hidden_size = vcfg.hidden_size # 1280
        self.num_heads = vcfg.num_attention_heads # 16
        self.intermediate_size = vcfg.intermediate_size # 5120
        self.layer_norm_eps = vcfg.layer_norm_eps # 1e-05
        self.num_layers = vcfg.num_hidden_layers # 32
        self.num_query_tokens = num_query_tokens
        self.cross_attention_freq = cross_attention_freq

        # 레이어별 cross-attn 적용 여부 미리 계산
        self._do_cross_map = [
            (cross_attention_freq > 0) and ((i + 1) % cross_attention_freq == 0)
            for i in range(self.num_layers)
        ]

        self.blocks = nn.ModuleList([
            QFormerEncoderBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                dropout=dropout,
                layer_norm_eps=self.layer_norm_eps,
            )
            for _ in range(self.num_layers)
        ])

        # CLIP-ViT-H-14 q-former에 init (Self-Attn + FFN)
        debugger = WeightDebugger()
        for i, blk in enumerate(self.blocks):
            clip_blk = vision.vision_model.encoder.layers[i]

            debugger.debug_load(blk.self_ln1, clip_blk.layer_norm1, "self_ln1", i)
            debugger.debug_load(blk.self_attn, clip_blk.self_attn, "self_attn", i)
            debugger.debug_load(blk.self_ln2, clip_blk.layer_norm2, "self_ln2", i)
            debugger.debug_load(blk.fc1, clip_blk.mlp.fc1, "fc1", i)
            debugger.debug_load(blk.fc2, clip_blk.mlp.fc2, "fc2", i)
        print(f"✅ Loaded CLIP pretrained weights for {self.num_layers} layers (Self-Attn + FFN only).")

        # load_state_dict는 weight만 저장 -> CLIP-ViT-H-14 메모리 해제
        del vision
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, x, fmri_feats, attn_mask=None):
        """
            x: [B, n_q+n_img, D]
            fmri_feats: [B, T, D]
            attn_mask: [n_q+n_img, n_q+n_img] (bool) or None
        """
        n_q = self.num_query_tokens
        for i, blk in enumerate(self.blocks):
            x = blk(
                x,
                fmri_feats=fmri_feats,
                attn_mask=attn_mask,
                n_q=n_q,
                do_cross=self._do_cross_map[i],
            )
        return x
    
class WeightDebugger:
    def __init__(self, atol=1e-6):
        self.atol = atol

    def debug_load(self, target_module, source_module, name, layer_idx):
        target_module.load_state_dict(source_module.state_dict())
        for k, v in source_module.state_dict().items():
            t = target_module.state_dict()[k]
            same = torch.allclose(t, v, atol=self.atol)
            print(f"[{layer_idx:02d}] {name}.{k:<28} | mean={v.mean():.6f}, std={v.std():.6f}, ok={same}")

# # 하나의 Q-former block(Self-Attention layer 1개 + (선택) Cross-Attention layer 1개 + FFN layer 1개)
class QFormerEncoderBlock(nn.Module):
    """
        - Self-Attention (query+image)
        - (선택) Cross-Attention (앞쪽 n_q query -> fmri_feats)
        - FFN (query+image)
        * 모듈은 항상 생성하고, 적용 여부는 forward의 do_cross로 제어 (init에 조건문 없음)
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        dropout = 0.1,
        layer_norm_eps = 1e-05,
    ):
        super().__init__()

        # (필수) Self-Attention layer 1개: (query+image) 
        self.self_ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention( embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_drop = nn.Dropout(dropout)

        # (선택) Cross-Attention layer 1개: query(앞 n_q 토큰) -> fmri_feats
        self.cross_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.cross_attn = nn.MultiheadAttention( embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_drop = nn.Dropout(dropout)

        # FFN layer 1개: (query+image)
        self.self_ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.ffn_drop = nn.Dropout(dropout)
        self.activation = nn.GELU()

    # 만약 mask가 None이면 inference 모드
    def forward(self, x, fmri_feats=None, attn_mask=None, n_q: int = None, do_cross: bool = False):
        """
            input:
                x: [B, 257+257, 1280]  (query + image concat)
                fmri_feats: [B, 20, 1280]
                attn_mask: [257+257, 257+257] bool or None
                n_q: int (number of query tokens)
            output:
                x: [B, 257+257, 1280]  
        """
        # (필수) Self-Attention layer 1개: (query+image) 
        residual = x
        x_norm = self.self_ln1(x)
        x_sa, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = residual + self.self_drop(x_sa)

        # (선택) Cross-Attention layer 1개: query(앞 n_q 토큰) -> fmri_feats
        if do_cross:
            q = x[:, :n_q, :]   # [B, n_q, D]
            kv = fmri_feats     # [B, T, D]
            q_res = q
            q_norm = self.cross_ln(q)
            q_ca, _ = self.cross_attn(q_norm, kv, kv, need_weights=False)
            q = q_res + self.cross_drop(q_ca)
            x = torch.cat([q, x[:, n_q:, :]], dim=1)

        # (필수) FFN layer 1개: (query+image)
        residual = x
        x_norm = self.self_ln2(x)
        x_ffn = self.fc2(self.ffn_drop(self.activation(self.fc1(x_norm))))
        x = residual + self.ffn_drop(x_ffn)
        return x

class Blip2FICLoss(nn.Module):
    """
        최소 구현 ITC(InfoNCE): query->image, image->query CE 평균
        - 입력 임베딩은 이미 L2 정규화되었다고 가정
        - learnable temperature만 선택적으로 제공
    """
    """
        FIC(Fused-CLS Matching) = BCEWithLogitsLoss 
            - 입력:
                logits: [2B]
                labels: [2B] (0/1)
            - 출력:
                loss (scalar)
    """
    def __init__(self, temperature_init: float = 0.07, learnable_temperature: bool = True):
        super().__init__()
        init_logit_scale = math.log(1.0 / temperature_init)  # log(1/τ)
        self.logit_scale = nn.Parameter(
            torch.tensor(init_logit_scale, dtype=torch.float32),
            requires_grad=learnable_temperature
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, query_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        # query_emb, image_emb: [B, D], 이미 L2 정규화된 상태
        B = query_emb.size(0)

        scale = self.logit_scale.exp().clamp(max=100.0)  # 안정성 위해 살짝 상한

        # cosine similarity(normalized dot product)
        logits_q2i = scale * (query_emb @ image_emb.t())  # [B, B]
        logits_i2q = scale * (image_emb @ query_emb.t())  # [B, B]

        # 정답 index: 주 대각선이 정답이라 각 행의 열을 0,1,2,...,B-1로 설정
        target = torch.arange(B, device=query_emb.device) # [0,1,2,...,B-1]
        # ce loss: 주 대각선이 분자이고 나머지가 분모
        loss = 0.5 * (self.ce(logits_q2i, target) + self.ce(logits_i2q, target)) # scalar
        return loss


class Blip2FIMLoss(nn.Module):
    """
        FIM(Fused-CLS Matching) = BCEWithLogitsLoss 
        - 입력:
            logits: [2B]
            labels: [2B] (0/1)
        - 출력:
            loss (scalar)
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, labels) -> torch.Tensor:
        logits = logits.view(-1)
        labels = labels.view(-1).to(dtype=logits.dtype)

        # scalar
        return F.binary_cross_entropy_with_logits(logits, labels, reduction=self.reduction) 
    

def get_model(args):
    #### 학습하는 모델을 담을 객체 ####
    connectimind2 = ConnecToMind2()

    #### Connectome-Transformer ####
    connectimind2.connectome_transformer = ConnectomeTransformer()

    #### fMRI-Image Aligner (FIC + FIM) ####
    connectimind2.q_former = FMRIImageAligner(
        num_query_tokens=257,
        hidden_size=1280,
        num_heads=16,
        num_layers=32,
        cross_attention_freq=2,
        intermediate_size=5120,
        dropout=0.1,
        layer_norm_eps=1e-12,
    )

    #### StableUnCLIP pipeline (SDXL v2.1) 추가 ####
    unclip_pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip",torch_dtype=torch.float16)

    models = {
        "connectimind2": connectimind2,
        "unclip_pipeline": unclip_pipeline,
    }

    return models
                                                  






