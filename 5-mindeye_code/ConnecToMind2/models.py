import os
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

class ConnectomeTransformer(nn.Module):
    def __init__(self, seq_len=20, input_dim=2056, embed_dim=1024, output_dim=257,  nhead=8, num_layers=8, is_position=False, is_fc=False, fc_matrix_path=""):
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
            x.shape [batch_size, roi개수, (voxel개수+padding)] -> [batch_size, roi개수, 1024]
        '''
        #### region-level fMRI embeddings ####
        x = torch.einsum("btd,tdh->bth", x, self.linear1_weight) # [B, roi개수, (voxel개수+padding)] -> [B, roi개수, 1024] - 각 roi마다 linear layer
        x = self.layernorm1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        #### Connectome-Transformer ####
        x = x + self.pos_embedding # positional embedding
        x = self.transformer_encoder(x)  # [B, 20, 1024] -> [B, 20, 1024]

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
    fMRI ↔ Image 정렬 (FIA) + 멀티모달 매칭 (FIM) 모델

    input:
        fmri_emb: [B, 20, 1024]   (fMRI embedding)
        image_emb: [B, 257, 1024]  (ViT-H/14 ip-adapter embedding; idx0 = CLS/global)

    내부 동작:
      FIA branch:
        query+image [B,257+257,768], mask [257+257,257+257] -> q-former -> fia_query_embedding [B,257,768], fia_image_embedding [B,257,768]
      FIM branch:
        query+image [2B,257+257,768], mask [257+257,257+257] -> q-former -> fim_logits [2B]

    output:
        {
            "fia_query_embedding" [B,257,768]
            "fia_image_embedding" [B,257,768]
            "fim_logits" [2B]
            "fim_labels" [2B]
        }
    """
    def __init__(
        self,
        num_query_tokens=257,     # IP-Adapter-Plus ViT-H/14 토큰 개수
        hidden_size=768,         # BLIP-2 default
        num_heads=12,            # BLIP-2 default
        num_layers=12,           # BLIP-2 default
        cross_attention_freq=2,  # BLIP-2 default
        intermediate_size=3072,  # BLIP-2 default (=768*4)
        dropout=0.1,             # BLIP-2 default
        layer_norm_eps=1e-12,    # BLIP-2 default
        initializer_range=0.02,  # BLIP-2/BERT init
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens
        self.initializer_range = initializer_range

        #### projection ####
        self.fmri_proj = nn.Linear(1024, hidden_size)   # [B,20,1024] -> [B,20,768]
        self.image_proj = nn.Linear(1024, hidden_size)   # [B,257,1024] -> [B,257,768]

        #### learnable query tokens (shared across ITC / ITM) ####
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size)) # [1,257,768] -> broadcast to [B,257,768]

        #### q_former ####
        self.q_former = QFormerEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            cross_attention_freq=cross_attention_freq,
            num_query_tokens=num_query_tokens,
        )

        #### heads ####
        # FIM classifier head
        self.itm_classifier = nn.Linear(hidden_size, 1)

        #### losses ####
        # FIA: align loss
        self.fia_loss_fn = nn.MSELoss()
        # FIM: positive match + negative match
        self.fim_loss_fn = Blip2FCMLoss()

        #### init weights ####
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # BLIP-2 / BERT style init
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def build_mask(self, kind, n_q, n_t, device):
        """
            mask: [L, L] bool 행렬 (True=차단, False=허용)
            if fim : 전범위 양방향
            if fia : Q와 T 상호 차단(블록 대각)
        """
        # nn.MultiheadAttention은 True:무시 & False:실행 -> zero로 초기화 = 모두 허용
        L = n_q + n_t
        mask = torch.zeros(L, L, dtype=torch.bool, device=device)

        if kind == "fim":
            return mask

        if kind == "fia":
            mask[:n_q, n_q:] = True                        # Q -> T 차단
            mask[n_q:, :n_q] = True                         # T -> Q 차단
            return mask

    def forward(self, args, fmri_emb, image_emb):
        """
            fmri_emb: [B, 20, 1024]   (fMRI embedding)
            image_emb: [B, 257, 1024]  (ViT-H/14 ip-adapter embedding; idx0 = CLS/global)   
            returns: dict
                {   
                    "fia_query_embedding" [B,257,768]
                    "fia_image_embedding" [B,257,768]
                    "fim_logits" [2B]
                    "fim_labels" [2B]
                }
        """
        B = fmri_emb.size(0)
        # query tokens
        query_tokens = self.query_tokens.expand(B, -1, -1)  # [B,257,768]

        # project inputs -> hidden_size
        fmri_feats = self.fmri_proj(fmri_emb)   # [B,20,768]
        image_feats = self.image_proj(image_emb)   # [B,257,768]

        n_q = query_tokens.size(1)   # 보통 257
        n_t = image_feats.size(1)    # 보통 257

        #### FIA branch ####
        q_i_fia = torch.cat([query_tokens, image_feats], dim=1) # [B,257+257,768]
        # q-former에 넣을 mask들 생성
        fia_attn_mask = self.build_mask("fia", n_q, n_t, device=args.device)  # [257+257,257+257] bool
        # q-former(self attention with all + cross-attn between only query and fMRI + FFN with all)
        q_i_hidden_fia = self.q_former(q_i_fia, fmri_feats, fia_attn_mask)  # [B,514,768]

        #### FIM branch ####
        # negative를 위해 image embedding 섞기
        perm = torch.randperm(B, device=args.device)
        # 2B(positive+negative)로 확장
        query_tokens_2b = torch.cat([query_tokens, query_tokens], dim=0) # [2B,257,768]
        image_feats_2b = torch.cat([image_feats, image_feats[perm]], dim=0) # [2B,257,768]
        fmri_feats_2b  = torch.cat([fmri_feats, fmri_feats], dim=0) # [2B,20,768]
        fim_labels_2b = torch.cat([torch.ones(B,device=args.device), torch.zeros(B,device=args.device)], dim=0) # [2B]

        # q-former에 넣을 mask들 생성
        fim_attn_mask = self.build_mask("fim", n_q, n_t, device=args.device)  # [257+257,257+257] bool        
        # q-former(self attention with all + cross-attn between only query and fMRI + FFN with all)
        q_i_fim = torch.cat([query_tokens_2b, image_feats_2b], dim=1) # [2B,257+257,768]
        q_i_hidden_fim = self.q_former(q_i_fim, fmri_feats_2b, fim_attn_mask)  # [2B,514,768]

        # cls + logit linear layer(768,1)
        fused_cls = q_i_hidden_fim[:, self.num_query_tokens, :]  # cls token 뽑기 [2B,768]
        fim_logits = self.itm_classifier(fused_cls).squeeze(-1)  # logit계산(linear layer) [2B,768] -> [2B]

        reps = {
            "fia_query_embedding": q_i_hidden_fia[:, :n_q, :],  # [B,257,768]
            "fia_image_embedding": q_i_hidden_fia[:, n_q:, :],  # [B,257,768]
            "fim_logits": fim_logits,               # [2B]
            "fim_labels": fim_labels_2b,            # [2B]
        }
        return reps

    # compute_loss
    def compute_loss(self, representations):
        """
        representations: dict output from forward()
            must contain q_hidden, t_hidden, itm_logits
        itm_labels: [B] or [B,1] with {0,1}, optional
        """
        fia_query_embedding = representations["fia_query_embedding"]     # [B,257,768]
        fia_image_embedding = representations["fia_image_embedding"]     # [B,257,768]
        fim_logits = representations["fim_logits"] # [2B]
        fim_labels = representations["fim_labels"] # [2B]

        #### FIA MSE (fine-grained alignment of all tokens) ####
        loss_fia = self.fia_loss_fn(fia_query_embedding, fia_image_embedding) # scalar

        #### FIM BCE (binary match) ####
        loss_fim = self.fim_loss_fn(fim_logits, fim_labels)  # scalar

        return {
            "loss_fia": loss_fia, # scalar
            "loss_fim": loss_fim,  # scalar 
        }
    
    def inference(self, fmri_emb):
        """
            fmri_emb: [B, 20, 1024]   (fMRI embedding)
            returns: 
                fia_query_embedding: [B,257,768]
        """
        B = fmri_emb.size(0)
        # query tokens
        query_tokens = self.query_tokens.expand(B, -1, -1)  # [B,257,768]
        # project inputs -> hidden_size
        fmri_feats = self.fmri_proj(fmri_emb)   # [B,20,768]
        query_hidden = self.q_former(query_tokens, fmri_feats, attn_mask=None)  # [B, 257, 768]
        return query_hidden

class QFormerEncoder(nn.Module):
    """
        - QFormerEncoderBlock을 num_layers개 쌓음
        - cross_attention_freq 간격으로 cross-attn 블록 배치 (예: 2 => 2,4,6,...번째 레이어에서 cross-attn)
        - 입력: [B, 257+257, 768]  (concat of [queries, images]), fmri_feats: [B, 20, 768], attn_mask:  [257+257, 257+257] (train) or None (inference)
        - 출력: [B, 257+257, 768]
    """
    def __init__(
        self,
        num_layers= 12,
        hidden_size= 768,
        num_heads= 12,
        intermediate_size= 3072,
        dropout= 0.1,
        layer_norm_eps= 1e-12,
        cross_attention_freq= 2,
        num_query_tokens= 257,   # fused 앞쪽 query 토큰 개수
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_query_tokens = num_query_tokens

        blocks = []
        for i in range(num_layers):
            do_cross = (cross_attention_freq > 0) and ((i + 1) % cross_attention_freq == 0)
            blocks.append(
                QFormerEncoderBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                    do_cross_attention=do_cross,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, fmri_feats, attn_mask=None):
        """
            input:
                x: [B, 257+257, 768]  (concat of [queries, images])
                fmri_feats: [B, 20, 768]
                attn_mask:  [257+257, 257+257] (train) or None (inference)
            output:
                x: [B, 257+257, 768]
        """
        n_q = self.num_query_tokens
        x = x
        for blk in self.blocks:
            x = blk(x, fmri_feats=fmri_feats, attn_mask=attn_mask, n_q=n_q)
        return x

# 하나의 Q-former block(Self-Attention layer 1개 + (선택) Cross-Attention layer 1개 + FFN layer 1개)
class QFormerEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        do_cross_attention: bool = False,
    ):
        super().__init__()
        self.do_cross_attention = do_cross_attention # 짝수 번째 레이어에서만 True

        # (필수) Self-Attention layer 1개: (query+image) 
        self.self_ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_drop = nn.Dropout(dropout)

        # (선택) Cross-Attention layer 1개: query(앞 n_q 토큰) -> fmri_feats
        if self.do_cross_attention:
            self.cross_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.cross_drop = nn.Dropout(dropout)

        # FFN layer 1개: (query+image)
        self.self_ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.ffn_drop = nn.Dropout(dropout)
        self.activation = nn.GELU()

    # 만약 mask가 None이면 inference 모드
    def forward(self, x, fmri_feats=None, attn_mask=None, n_q: int = None):
        """
            input:
                x: [B, 257+257, 768]  (query + image concat)
                fmri_feats: [B, 20, 768]
                attn_mask: [257+257, 257+257] bool or None
                n_q: int (number of query tokens)
            output:
                x: [B, 257+257, 768]  
        """
        # (필수) Self-Attention layer 1개: (query+image) 
        residual = x
        x_norm = self.self_ln1(x)
        # nn.MultiheadAttention: attn_mask는 (257+257, 257+257) 가능 (bool: True=mask)
        x_sa, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = residual + self.self_drop(x_sa)

        # (선택) Cross-Attention layer 1개: query(앞 n_q 토큰) -> fmri_feats
        if self.do_cross_attention:
            q = x[:, :n_q, :]   # [B, n_q, 768]
            kv = fmri_feats     # [B, T, 768]
            q_residual = q
            q_norm = self.cross_ln(q)
            q_ca, _ = self.cross_attn(q_norm, kv, kv, need_weights=False)  # mask 필요시 추가 가능
            q = q_residual + self.cross_drop(q_ca)
            # 갱신한 query 부분을 합치기
            x[:, :n_q, :] = q # [B, n_q+image, 768]

        # (필수) FFN layer 1개: (query+image)
        residual = x
        x_norm = self.self_ln2(x)
        x_ffn = self.fc2(self.ffn_drop(self.activation(self.fc1(x_norm))))
        x = residual + self.ffn_drop(x_ffn)

        return x

class Blip2FCMLoss(nn.Module):
    """
    FCM(Fused-CLS Matching) = BCEWithLogitsLoss 
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
    

# class Blip2FICLoss(nn.Module):
#     """
#     contrastive ITC loss (BLIP-2 스타일) + query 축소
#     - q_hidden: [B,257,768]
#     - t_hidden: [B,257,768]
#     1) q_hidden을 learnable linear mixing으로 257 -> reduced_query_tokens(예:32)로 압축
#     2) text CLS (t_hidden[:,0,:])와 cosine sim
#     3) max over query_slots (32)
#     4) in-batch CLIP-style cross entropy
#     """

#     def __init__(
#         self,
#         in_tokens=257,
#         reduced_query_tokens=32,
#         hidden_size=768,
#         init_logit_scale=1/0.07,
#     ):
#         super().__init__()
#         self.in_tokens = in_tokens
#         self.reduced_query_tokens = reduced_query_tokens
#         self.hidden_size = hidden_size

#         # linear mixing matrix: [257,32]
#         self.reduce = nn.Parameter(torch.randn(in_tokens, reduced_query_tokens) * 0.01)

#         # learnable temperature
#         self.logit_scale = nn.Parameter(torch.log(torch.tensor(init_logit_scale, dtype=torch.float32)))

#     def forward(self, q_hidden, t_hidden):
#         """
#         q_hidden: [B, in_tokens, hidden_size]   e.g. [B,257,768]
#         i_hidden: [B, T, hidden_size]          e.g. [B,257,768]
#         returns: scalar contrastive loss
#         """
#         B, Tin, H = q_hidden.shape
#         # image CLS
#         image_cls = t_hidden[:, 0, :]    # [B,768]

#         # 257 -> 32
#         # q_hidden: [B,257,768] -> [B,768,257]
#         q_T = q_hidden.transpose(1, 2)                  # [B,768,257]
#         # matmul with [257,32] -> [B,768,32]
#         q_reduced = torch.matmul(q_T, self.reduce)      # [B,768,32]
#         # back to [B,32,768]
#         q_reduced = q_reduced.transpose(1, 2).contiguous()  # [B,32,768]

#         # normalize
#         q_norm = F.normalize(q_reduced, dim=-1)  # [B,32,768]
#         i_norm = F.normalize(image_cls, dim=-1)   # [B,768]

#         # cosine similarity for all pairs
#         # q_norm.unsqueeze(2): [B,32,1,768]
#         # t_norm.unsqueeze(0,0): [1,1,B,768]
#         sim_all = torch.sum(
#             q_norm.unsqueeze(2) * i_norm.unsqueeze(0).unsqueeze(0),
#             dim=-1
#         )  # [B,32,B]

#         # max over query slots -> [B,B]
#         sim_i2t = sim_all.max(dim=1).values           # [B,B]
#         sim_t2i = sim_i2t.t().contiguous()            # [B,B]

#         # temperature scaling
#         logit_scale = self.logit_scale.exp()
#         logits_i2t = sim_i2t * logit_scale
#         logits_t2i = sim_t2i * logit_scale

#         # symmetric CE
#         targets = torch.arange(B, device=q_hidden.device)
#         loss_i2t = F.cross_entropy(logits_i2t, targets)
#         loss_t2i = F.cross_entropy(logits_t2i, targets)
#         loss_contrastive = 0.5 * (loss_i2t + loss_t2i)

#         return loss_contrastive
    