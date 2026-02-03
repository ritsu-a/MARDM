import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import clip
import math
from functools import partial
from timm.models.vision_transformer import Mlp
from models.DiffMLPs import DiffMLPs_models
from utils.eval_utils import eval_decorator
from utils.train_utils import lengths_to_mask, uniform, get_mask_subset_prob, cosine_schedule

#################################################################################
#                                      MARDM                                    #
#################################################################################
class MARDM(nn.Module):
    def __init__(self, ae_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.2, clip_dim=512,
                 diffmlps_batch_mul=4, diffmlps_model='SiT-XL', cond_drop_prob=0.1,
                 clip_version='ViT-B/32', **kargs):
        super(MARDM, self).__init__()

        self.ae_dim = ae_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
            self.num_actions = kargs.get('num_actions', 1)
            self.encode_action = partial(F.one_hot, num_classes=self.num_actions)
        
        # Set up condition mode and cross-attention flag
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
            self.use_cross_attn = False
        elif self.cond_mode == 'audio':
            # For audio (whisper features), feature dimension is typically 512 for base model
            audio_dim = kargs.get('audio_dim', 512)
            self.cond_emb = nn.Linear(audio_dim, self.latent_dim)
            # Audio sequence embedding for cross-attention
            self.audio_seq_emb = nn.Linear(audio_dim, self.latent_dim)
            self.use_cross_attn = kargs.get('use_cross_attn', True)  # Enable cross-attention by default
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
            self.use_cross_attn = False
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
            self.use_cross_attn = False
        else:
            raise KeyError("Unsupported condition mode!!!")
        
        # --------------------------------------------------------------------------
        # MAR Tranformer
        print('Loading MARTransformer...')
        self.input_process = InputProcess(self.ae_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        # Create transformer blocks with cross-attention for audio mode
        self.MARTransformer = nn.ModuleList([
            MARTransBlock(self.latent_dim, num_heads, mlp_size=ff_size, drop_out=self.dropout, 
                         use_cross_attn=self.use_cross_attn) for _ in range(num_layers)
        ])

        self.mask_latent = nn.Parameter(torch.zeros(1, 1, self.ae_dim))

        self.apply(self.__init_weights)
        for block in self.MARTransformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)
        elif self.cond_mode == 'audio':
            print('Using audio (whisper) features as condition...')
            # No need to load CLIP for audio mode

        # --------------------------------------------------------------------------
        # DiffMLPs
        print('Loading DiffMLPs...')
        self.DiffMLPs = DiffMLPs_models[diffmlps_model](target_channels=self.ae_dim, z_channels=self.latent_dim)
        self.diffmlps_batch_mul = diffmlps_batch_mul

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)
        assert torch.cuda.is_available()
        clip.model.convert_weights(clip_model)

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, latents, cond, padding_mask, force_mask=False, mask=None, audio_seq=None, audio_mask=None):
        """
        Args:
            latents: motion latents [B, L, ae_dim]
            cond: condition vector [B, latent_dim] (for adaLN modulation)
            padding_mask: motion padding mask [B, L]
            force_mask: whether to force mask condition
            mask: optional mask
            audio_seq: audio feature sequence [B, T_audio, audio_dim] (for cross-attention)
            audio_mask: audio padding mask [B, T_audio]
        """
        # Process condition vector for adaLN modulation
        if cond.dim() == 1 or (cond.dim() == 2 and cond.shape[1] != self.latent_dim):
            # Raw condition, needs processing
            cond = self.mask_cond(cond, force_mask=force_mask)
            cond = self.cond_emb(cond)
        else:
            # Already processed condition vector
            cond = self.mask_cond(cond, force_mask=force_mask)
        
        # Verify latents shape: should be [B, L, ae_dim]
        if latents.dim() != 3:
            raise ValueError(f"Expected 3D latents tensor [B, L, ae_dim], got shape {latents.shape}")
        B, L, ae_dim = latents.shape
        
        # Verify padding_mask shape matches latents sequence length
        if padding_mask is not None:
            if padding_mask.dim() != 2:
                raise ValueError(f"Expected 2D padding_mask tensor [B, L], got shape {padding_mask.shape}")
            if padding_mask.size(0) != B:
                raise ValueError(f"padding_mask batch size ({padding_mask.size(0)}) does not match latents batch size ({B})")
            if padding_mask.size(1) != L:
                raise ValueError(f"padding_mask sequence length ({padding_mask.size(1)}) does not match latents sequence length ({L}). latents.shape={latents.shape}, padding_mask.shape={padding_mask.shape}")
        
        x = self.input_process(latents)
        x = self.position_enc(x)
        x = x.permute(1, 0, 2)  # [L, B, latent_dim]

        # Process audio sequence for cross-attention
        audio_seq_processed = None
        if self.use_cross_attn and audio_seq is not None:
            # audio_seq: [B, T_audio, audio_dim] -> [B, T_audio, latent_dim]
            audio_seq_processed = self.audio_seq_emb(audio_seq)
            # Convert to [T_audio, B, latent_dim] for consistency
            audio_seq_processed = audio_seq_processed.permute(1, 0, 2)

        for block in self.MARTransformer:
            if self.use_cross_attn and audio_seq_processed is not None:
                x = block(x, cond, padding_mask, audio_seq=audio_seq_processed, audio_mask=audio_mask)
            else:
                x = block(x, cond, padding_mask)
        return x

    def forward_loss(self, latents, y, m_lens):
        latents = latents.permute(0, 2, 1)
        b, l, d = latents.shape
        device = latents.device

        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(non_pad_mask.unsqueeze(-1), latents, torch.zeros_like(latents))

        target = latents.clone().detach()
        input = latents.clone()

        force_mask = False
        audio_seq = None
        audio_mask = None
        
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'audio':
            # y can be either:
            # - [batch_size, audio_dim] - mean pooled (for backward compatibility)
            # - [batch_size, T_audio, audio_dim] - full sequence (for cross-attention)
            y_tensor = y.to(device).float() if torch.is_tensor(y) else torch.from_numpy(y).to(device).float()
            
            if len(y_tensor.shape) == 2:
                # Mean pooled: [batch_size, audio_dim]
                cond_vector = self.cond_emb(y_tensor)
            elif len(y_tensor.shape) == 3:
                # Full sequence: [batch_size, T_audio, audio_dim]
                # Use mean for adaLN modulation, keep sequence for cross-attention
                cond_vector = self.cond_emb(y_tensor.mean(dim=1))  # [batch_size, latent_dim]
                audio_seq = y_tensor  # [batch_size, T_audio, audio_dim]
                # Create audio mask (assuming no padding for now, can be extended)
                audio_mask = None  # TODO: add audio mask if needed
            else:
                raise ValueError(f"Unexpected audio feature shape: {y_tensor.shape}")
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        rand_time = uniform((b,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_masked = (l * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((b, l), device=device).argsort(dim=-1)
        mask = batch_randperm < num_masked.unsqueeze(-1)
        mask &= non_pad_mask
        mask_rlatents = get_mask_subset_prob(mask, 0.1)
        rand_latents = torch.randn_like(input)
        input = torch.where(mask_rlatents.unsqueeze(-1), rand_latents, input)
        mask_mlatents = get_mask_subset_prob(mask & ~mask_rlatents, 0.88)
        input = torch.where(mask_mlatents.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), input)

        z = self.forward(input, cond_vector, ~non_pad_mask, force_mask, audio_seq=audio_seq, audio_mask=audio_mask)
        target = target.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        z = z.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        mask = mask.reshape(b * l).repeat(self.diffmlps_batch_mul)
        target = target[mask]
        z = z[mask]
        loss = self.DiffMLPs(z=z, target=target)

        return loss

    def forward_with_CFG(self, latents, cond_vector, padding_mask, cfg=3, mask=None, force_mask=False, audio_seq=None, audio_mask=None):
        if force_mask:
            return self.forward(latents, cond_vector, padding_mask, force_mask=True, mask=None, audio_seq=None, audio_mask=None)

        logits = self.forward(latents, cond_vector, padding_mask, mask=None, audio_seq=audio_seq, audio_mask=audio_mask)
        if cfg != 1:
            aux_logits = self.forward(latents, cond_vector, padding_mask, force_mask=True, mask=None, audio_seq=None, audio_mask=None)
            mixed_logits = torch.cat([logits, aux_logits], dim=0)
        else:
            mixed_logits = logits
        b, l, d = mixed_logits.size()
        if mask is not None:
            mask2 = torch.cat([mask, mask], dim=0).reshape(b * l)
            mixed_logits = (mixed_logits.reshape(b * l, d))[mask2]
        else:
            mixed_logits = mixed_logits.reshape(b * l, d)
        output = self.DiffMLPs.sample(mixed_logits, 1, cfg)
        if cfg != 1:
            scaled_logits, _ = output.chunk(2, dim=0)
        else:
            scaled_logits = output
        if mask is not None:
            latents = latents.reshape(b//2 * l, self.ae_dim)
            latents[mask.reshape(b//2 * l)] = scaled_logits
            scaled_logits = latents.reshape(b//2, l, self.ae_dim)

        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 force_mask=False
                 ):
        device = next(self.parameters()).device
        l = max(m_lens)
        b = len(m_lens)

        audio_seq = None
        audio_mask = None
        
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'audio':
            conds_tensor = conds.to(device).float() if torch.is_tensor(conds) else torch.from_numpy(conds).to(device).float()
            if len(conds_tensor.shape) == 2:
                cond_vector = self.cond_emb(conds_tensor)
            elif len(conds_tensor.shape) == 3:
                cond_vector = self.cond_emb(conds_tensor.mean(dim=1))
                audio_seq = conds_tensor
            else:
                raise ValueError(f"Unexpected audio feature shape: {conds_tensor.shape}")
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, l)

        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(b, l, self.ae_dim).to(device),
                          self.mask_latent.repeat(b, l, 1))
        masked_rand_schedule = torch.where(padding_mask, 1e5, torch.rand_like(padding_mask, dtype=torch.float))

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            rand_mask_prob = cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))

            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), latents)
            logits = self.forward_with_CFG(latents, cond_vector=cond_vector, padding_mask=padding_mask,
                                                  cfg=cond_scale, mask=is_mask, force_mask=force_mask,
                                                  audio_seq=audio_seq, audio_mask=audio_mask)
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)

            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
        return latents.permute(0,2,1)

    @torch.no_grad()
    @eval_decorator
    def edit(self,
             conds,
             latents,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             force_mask=False,
             edit_mask=None,
             padding_mask=None,
             ):

        device = next(self.parameters()).device
        l = latents.shape[-1]

        audio_seq = None
        audio_mask = None
        
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'audio':
            conds_tensor = conds.to(device).float() if torch.is_tensor(conds) else torch.from_numpy(conds).to(device).float()
            if len(conds_tensor.shape) == 2:
                cond_vector = self.cond_emb(conds_tensor)
            elif len(conds_tensor.shape) == 3:
                cond_vector = self.cond_emb(conds_tensor.mean(dim=1))
                audio_seq = conds_tensor
            else:
                raise ValueError(f"Unexpected audio feature shape: {conds_tensor.shape}")
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask == None:
            padding_mask = ~lengths_to_mask(m_lens, l)

        if edit_mask == None:
            mask_free = True
            latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(latents.shape[0], l, self.ae_dim).to(device),
                                  latents.permute(0, 2, 1))
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            masked_rand_schedule = torch.where(edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(latents.shape[0], l, self.ae_dim).to(device),
                              latents.permute(0, 2, 1))
            latents = torch.where(edit_mask.unsqueeze(-1),
                              self.mask_latent.repeat(latents.shape[0], l, 1), latents)
            masked_rand_schedule = torch.where(edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5)

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            rand_mask_prob = 0.16 if mask_free else cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(
                dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))

            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(latents.shape[0], latents.shape[1], 1), latents)
            logits = self.forward_with_CFG(latents, cond_vector=cond_vector, padding_mask=padding_mask,
                                                  cfg=cond_scale, mask=is_mask, force_mask=force_mask,
                                                  audio_seq=audio_seq, audio_mask=audio_mask)
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)

            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        latents = torch.where(edit_mask.unsqueeze(-1), latents, latents)
        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
        return latents.permute(0,2,1)

#################################################################################
#                                     MARDM Zoos                                #
#################################################################################
def mardm_ddpm_xl(**kwargs):
    return MARDM(latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16, dropout=0.2, clip_dim=512,
                 diffmlps_model="DDPM-XL", diffmlps_batch_mul=4, cond_drop_prob=0.1, **kwargs)
def mardm_sit_xl(**kwargs):
    return MARDM(latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16, dropout=0.2, clip_dim=512,
                 diffmlps_model="SiT-XL", diffmlps_batch_mul=4, cond_drop_prob=0.1, **kwargs)

MARDM_models = {
    'MARDM-DDPM-XL': mardm_ddpm_xl, 'MARDM-SiT-XL': mardm_sit_xl,
}

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
def modulate_here(x, shift, scale):
    """
    Modulate x with shift and scale using adaLN.
    Supports both formats:
    - Batch-first: x [B, T, C], shift/scale [B, C] -> use unsqueeze(1)
    - Sequence-first: x [T, B, C], shift/scale [B, C] -> use unsqueeze(0)
    """
    # Detect format by checking dimensions
    if x.dim() == 3:
        # Priority: check batch-first first (most common case)
        # Batch-first: [B, T, C] where B == shift.size(0)
        if x.size(0) == shift.size(0):
            # Batch-first format: [B, T, C], shift/scale [B, C]
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        # Sequence-first: [T, B, C] where B == shift.size(0) and T != shift.size(0)
        elif x.size(1) == shift.size(0) and x.size(0) != shift.size(0):
            # Sequence-first format: [T, B, C], shift/scale [B, C]
            return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        else:
            raise ValueError(f"Dimension mismatch: x shape {x.shape}, shift shape {shift.shape}, scale shape {scale.shape}")
    else:
        # Fallback to batch-first
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute((1, 0, 2))
        x = self.poseEmbedding(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.2):
        super().__init__()
        assert embed_dim % 8 == 0
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, mask):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            # mask should be [B, T] where T matches the sequence length
            # att is [B, n_head, T, T]
            # Expand mask to [B, 1, 1, T] to match att dimensions
            if mask.dim() == 2:
                # mask is [B, T], expand to [B, 1, 1, T]
                # Verify mask shape matches x sequence length
                if mask.size(1) != T:
                    raise ValueError(f"Mask sequence length ({mask.size(1)}) does not match input sequence length ({T}). mask.shape={mask.shape}, x.shape={x.shape}")
                mask_expanded = mask[:, None, None, :]
            elif mask.dim() == 1:
                # mask is [T], expand to [1, 1, 1, T]
                if mask.size(0) != T:
                    raise ValueError(f"Mask sequence length ({mask.size(0)}) does not match input sequence length ({T}). mask.shape={mask.shape}, x.shape={x.shape}")
                mask_expanded = mask[None, None, None, :]
            else:
                mask_expanded = mask
            att = att.masked_fill(mask_expanded != 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y


class CrossAttention(nn.Module):
    """
    Cross-attention layer for motion tokens to attend to audio features
    Query: motion tokens (x)
    Key & Value: audio features (audio_seq)
    """
    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.2):
        super().__init__()
        assert embed_dim % n_head == 0
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, audio_seq, audio_mask=None):
        """
        Args:
            x: motion tokens [B, T_motion, C]
            audio_seq: audio features [B, T_audio, C]
            audio_mask: audio padding mask [B, T_audio], True for padding positions
        Returns:
            output: [B, T_motion, C]
        """
        # Ensure correct dimensions: both should be [B, T, C] format (batch-first)
        if x.dim() != 3 or audio_seq.dim() != 3:
            raise ValueError(f"Expected 3D tensors, got x.shape={x.shape}, audio_seq.shape={audio_seq.shape}")
        
        # x should be [B, T_motion, C], audio_seq should be [B, T_audio, C]
        # Check if we need to permute (sequence-first -> batch-first)
        # Typical: T_motion=45, T_audio=150, B=16, C=1024
        # If first dim > second dim, it's likely sequence-first format
        if x.size(0) > x.size(1):
            # x is [T_motion, B, C], convert to [B, T_motion, C]
            x = x.permute(1, 0, 2)
        if audio_seq.size(0) > audio_seq.size(1):
            # audio_seq is [T_audio, B, C], convert to [B, T_audio, C]
            audio_seq = audio_seq.permute(1, 0, 2)
        
        # Now both should be batch-first: [B, T, C]
        B, T_motion, C = x.size()
        B_audio, T_audio, C_audio = audio_seq.size()
        
        # Verify batch size and feature dimension match
        if B != B_audio:
            raise ValueError(f"Batch size mismatch: x batch={B}, audio_seq batch={B_audio}. x.shape={x.shape}, audio_seq.shape={audio_seq.shape}")
        if C != C_audio:
            raise ValueError(f"Feature dimension mismatch: x C={C}, audio_seq C={C_audio}. x.shape={x.shape}, audio_seq.shape={audio_seq.shape}")

        # Query from motion tokens: [B, T_motion, C] -> [B, n_head, T_motion, head_dim]
        q = self.query(x).view(B, T_motion, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Key and Value from audio features: [B, T_audio, C] -> [B, n_head, T_audio, head_dim]
        k = self.key(audio_seq).view(B, T_audio, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(audio_seq).view(B, T_audio, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute attention: [B, n_head, T_motion, T_audio]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply audio mask if provided
        if audio_mask is not None:
            audio_mask = audio_mask[:, None, None, :]  # [B, 1, 1, T_audio]
            att = att.masked_fill(audio_mask != 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Apply attention to values
        y = att @ v  # [B, n_head, T_motion, C // n_head]
        y = y.transpose(1, 2).contiguous().view(B, T_motion, C)

        y = self.resid_drop(self.proj(y))
        return y


class MARTransBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_size=1024, drop_out=0.2, use_cross_attn=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, drop_out_rate=drop_out)
        
        # Cross-attention for audio features
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads, drop_out_rate=drop_out)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = mlp_size
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, padding_mask=None, audio_seq=None, audio_mask=None):
        """
        Args:
            x: motion tokens [T_motion, B, C] (sequence-first format)
            c: condition vector [B, C] for adaLN
            padding_mask: [B, T_motion] motion padding mask
            audio_seq: audio features [T_audio, B, C] (sequence-first format) or None
            audio_mask: audio padding mask [B, T_audio] or None
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention
        # x should be [T_motion, B, C] (sequence-first) from forward method
        # padding_mask is [B, T_motion] (batch-first)
        # Convert to batch-first for attention, then convert back
        # Check if x is batch-first or sequence-first
        if x.size(0) == padding_mask.size(0) and x.size(1) != padding_mask.size(0):
            # x is batch-first [B, T_motion, C]
            x_batch = x
            x_was_batch_first = True
        elif x.size(1) == padding_mask.size(0):
            # x is sequence-first [T_motion, B, C]
            x_batch = x.permute(1, 0, 2)  # [T_motion, B, C] -> [B, T_motion, C]
            x_was_batch_first = False
        else:
            raise ValueError(f"Cannot determine x format. x.shape={x.shape}, padding_mask.shape={padding_mask.shape}")
        
        x_norm = self.norm1(x_batch)  # [B, T_motion, C]
        x_modulated = modulate_here(x_norm, shift_msa, scale_msa)  # [B, T_motion, C]
        # padding_mask is [B, T_motion], verify it matches x_batch sequence length
        if padding_mask is not None:
            if padding_mask.size(1) != x_batch.size(1):
                raise ValueError(f"padding_mask sequence length ({padding_mask.size(1)}) does not match x sequence length ({x_batch.size(1)}). padding_mask.shape={padding_mask.shape}, x_batch.shape={x_batch.shape}, x.shape={x.shape}")
        attn_out = self.attn(x_modulated, mask=padding_mask)  # [B, T_motion, C]
        
        # Convert back to original format
        if x_was_batch_first:
            # x was batch-first, keep batch-first
            attn_out_seq = attn_out
            gate_for_attn = gate_msa.unsqueeze(1)  # [B, C] -> [B, 1, C] for batch-first format
        else:
            # x was sequence-first, convert back
            attn_out_seq = attn_out.permute(1, 0, 2)  # [B, T_motion, C] -> [T_motion, B, C]
            gate_for_attn = gate_msa.unsqueeze(0)  # [B, C] -> [1, B, C] for sequence-first format
        x = x + gate_for_attn * attn_out_seq
        
        # Cross-attention to audio features
        if self.use_cross_attn and audio_seq is not None:
            # x format depends on what was used in self-attention
            # audio_seq is [T_audio, B, C] (sequence-first)
            # gate_msa is [B, C]
            
            if x_was_batch_first:
                # x is batch-first [B, T_motion, C]
                # audio_seq: [T_audio, B, C] -> [B, T_audio, C]
                audio_seq_batch_first = audio_seq.permute(1, 0, 2)  # [150, 16, 1024] -> [16, 150, 1024]
                
                # Apply cross-attention (expects batch-first format)
                x_cross_batch_first = self.cross_attn(
                    modulate_here(self.norm_cross(x), shift_msa, scale_msa), 
                    audio_seq_batch_first, 
                    audio_mask
                )
                # x_cross_batch_first is [B, T_motion, C]
                
                # Apply gate: gate_msa is [B, C]
                gate_for_batch = gate_msa.unsqueeze(1)  # [B, C] -> [B, 1, C] for batch-first format
                x = x + gate_for_batch * x_cross_batch_first
            else:
                # x is sequence-first [T_motion, B, C]
                # Convert to batch-first format for cross-attention
                x_batch_first = x.permute(1, 0, 2)  # [45, 16, 1024] -> [16, 45, 1024]
                # audio_seq: [T_audio, B, C] -> [B, T_audio, C]
                audio_seq_batch_first = audio_seq.permute(1, 0, 2)  # [150, 16, 1024] -> [16, 150, 1024]
                
                # Apply cross-attention (expects batch-first format)
                x_cross_batch_first = self.cross_attn(
                    modulate_here(self.norm_cross(x_batch_first), shift_msa, scale_msa), 
                    audio_seq_batch_first, 
                    audio_mask
                )
                # x_cross_batch_first is [B, T_motion, C]
                
                # Convert back to sequence-first format: [B, T_motion, C] -> [T_motion, B, C]
                x_cross = x_cross_batch_first.permute(1, 0, 2)  # [16, 45, 1024] -> [45, 16, 1024]
                
                # Apply gate: gate_msa is [B, C]
                gate_for_seq = gate_msa.unsqueeze(0)  # [B, C] -> [1, B, C] for sequence-first format
                x = x + gate_for_seq * x_cross
        
        # MLP
        # x format depends on what was used in self-attention
        # Use the same format as self-attention output
        if x_was_batch_first:
            # x is batch-first [B, T_motion, C]
            x_norm_mlp = self.norm2(x)  # [B, T_motion, C]
            x_modulated_mlp = modulate_here(x_norm_mlp, shift_mlp, scale_mlp)  # [B, T_motion, C]
            mlp_out = self.mlp(x_modulated_mlp)  # [B, T_motion, C]
            gate_for_mlp = gate_mlp.unsqueeze(1)  # [B, C] -> [B, 1, C] for batch-first format
            x = x + gate_for_mlp * mlp_out
        else:
            # x is sequence-first [T_motion, B, C]
            x_batch_mlp = x.permute(1, 0, 2)  # [T_motion, B, C] -> [B, T_motion, C]
            x_norm_mlp = self.norm2(x_batch_mlp)  # [B, T_motion, C]
            x_modulated_mlp = modulate_here(x_norm_mlp, shift_mlp, scale_mlp)  # [B, T_motion, C]
            mlp_out = self.mlp(x_modulated_mlp)  # [B, T_motion, C]
            mlp_out_seq = mlp_out.permute(1, 0, 2)  # [B, T_motion, C] -> [T_motion, B, C]
            gate_for_mlp = gate_mlp.unsqueeze(0)  # [B, C] -> [1, B, C] for sequence-first format
            x = x + gate_for_mlp * mlp_out_seq
        return x
