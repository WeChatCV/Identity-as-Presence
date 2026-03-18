# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import glob
import json
import math
import os
import types
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from torch import nn

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    usp_attn_forward, xFuserLongContextAttention)
from ..utils import cfg_skip
from .cache_utils import TeaCache

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    major, minor = torch.cuda.get_device_capability(0)
    if f"{major}.{minor}" == "8.0":
        from sageattention_sm80 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "8.6":
        from sageattention_sm86 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "8.9":
        from sageattention_sm89 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "9.0":
        from sageattention_sm90 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif major>9:
        from sageattention_sm120 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
except:
    try:
        from sageattention import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    except:
        sageattn = None
        SAGE_ATTENTION_AVAILABLE = False


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    attention_type=None,
    custom_attn_mask=None
):
    attention_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION") if attention_type is None else attention_type
    if torch.is_grad_enabled() and attention_type == "SAGE_ATTENTION":
        attention_type = "FLASH_ATTENTION"
    if attention_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        out = sageattn(
            q, k, v, attn_mask=attn_mask, tensor_layout="NHD", is_causal=causal, dropout_p=dropout_p)

    elif attention_type == "FLASH_ATTENTION" and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
    return out


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000, freqs_scaling=1.0):
    assert dim % 2 == 0
    pos =  torch.arange(max_seq_len)
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    freqs = freqs_scaling * freqs
    freqs = torch.outer(pos, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

# modified from https://github.com/thu-ml/RIFLEx/blob/main/riflex_utils.py
@amp.autocast(enabled=False)
def get_1d_rotary_pos_embed_riflex(
    pos: Union[np.ndarray, int],
    dim: int,
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
    L_test_scale: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = 1.0 / torch.pow(theta,
        torch.arange(0, dim, 2).to(torch.float64).div(dim))

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===
    if L_test_scale is not None:
        freqs[k-1] = freqs[k-1] / L_test_scale

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis

# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs, offsets=None):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return rope_apply_3d(x, grid_sizes, freqs, offsets=offsets)
    else:
        return rope_apply_1d(x, grid_sizes, freqs, offsets=offsets)

@amp.autocast(enabled=False)
def rope_apply_3d(x, grid_sizes, freqs, offsets=None):

    n, c = x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    if offsets is None:
        f_offset = 0
        h_offset = 0
        w_offset = 0
    else:
        f_offset = offsets[0]
        h_offset = offsets[1]
        w_offset = offsets[2]
        
    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))

        freqs_i = torch.cat([
            freqs[0][f_offset:f_offset+f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][h_offset:h_offset+h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][w_offset:w_offset+w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)

@amp.autocast(enabled=False)
def rope_apply_1d(x, grid_sizes, freqs, offsets=None):
    n, c = x.size(2), x.size(3) // 2 ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"
    
    # loop over samples
    output = []
    for i, (l, ) in enumerate(grid_sizes.tolist()):
        seq_len = l
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2)) # [l n d//2]
        x_i_rope = x_i[:, :, :c_rope] * freqs[:seq_len, None, :]  # [L, N, c_rope]
        x_i_passthrough = x_i[:, :, c_rope:]  # untouched dims
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()


def rope_apply_qk(q, k, grid_sizes, freqs, offsets=None):
    q = rope_apply(q, grid_sizes, freqs, offsets=offsets)
    k = rope_apply(k, grid_sizes, freqs, offsets=offsets)
    return q, k


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply_prime(x, grid_sizes, grid_sizes_prime, freqs, offsets=None):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return rope_apply_prime_3d(x, grid_sizes, grid_sizes_prime, freqs, offsets=offsets)
    else:
        return rope_apply_prime_1d(x, grid_sizes, grid_sizes_prime, freqs, offsets=offsets)


def rope_apply_prime_3d(x_prime, grid_sizes_ori, grid_sizes_prime, freqs, offsets=None):
    """
    Apply RoPE to extra tokens, position encoding continues from video position
    
    Args:
        x_prime: extra tokens, shape [B, L_extra, n, c]
        grid_sizes_ori: video grid sizes, shape [B, 3] containing (F, H, W)
        grid_sizes_prime: extra grid sizes, shape [B, 3]
        freqs: RoPE frequencies
    
    Returns:
        extra tokens with position encoding applied
    """
    n, c = x_prime.size(2), x_prime.size(3) // 2
    
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # loop over samples
    output = []
    for i, ((f, h, w), (f_prime, h_prime, w_prime)) in enumerate(zip(grid_sizes_ori.tolist(), grid_sizes_prime.tolist())):
        seq_len_prime = x_prime.size(1)
        assert seq_len_prime == f_prime * h_prime * w_prime
        
        if offsets is None:
            f_offset = f
            h_offset = h
            w_offset = w
        else:
            f_offset = offsets[0]
            h_offset = offsets[1]
            w_offset = offsets[2]
        
        # build position encoding for each extra token
        x_i = torch.view_as_complex(x_prime[i, :seq_len_prime].to(torch.float32).reshape(
            seq_len_prime, n, -1, 2))
        
        freqs_i = torch.cat([
            freqs[0][f_offset:f_offset+f_prime].view(f_prime, 1, 1, -1).expand(f_prime, h_prime, w_prime, -1),  # F dimension: start from f
            freqs[1][h_offset:h_offset+h_prime].view(1, h_prime, 1, -1).expand(f_prime, h_prime, w_prime, -1),  # H dimension: start from h
            freqs[2][w_offset:w_offset+w_prime].view(1, 1, w_prime, -1).expand(f_prime, h_prime, w_prime, -1)   # W dimension: start from w
            ], dim=-1).reshape(seq_len_prime, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x_prime[i, seq_len_prime:]])
        
        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x_prime.dtype)


def rope_apply_prime_1d(x_prime, grid_sizes_ori, grid_sizes_prime, freqs, offsets=None):
    n, c = x_prime.size(2), x_prime.size(3) // 2 ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"
    
    # loop over samples
    output = []
    for i, ((l, ),(l_prime, )) in enumerate(zip(grid_sizes_ori.tolist(), grid_sizes_prime.tolist())):
        if offsets != None:
            seq_len = offsets[0]
        else:
            seq_len = l
        seq_len_prime = x_prime.size(1)
        # precompute multipliers
        x_i = torch.view_as_complex(x_prime[i, :seq_len_prime].to(torch.float64).reshape(
            seq_len_prime, n, -1, 2)) # [l n d//2]
            
        x_i_rope = x_i[:, :, :c_rope] * freqs[seq_len:seq_len+seq_len_prime, None, :]  # [L, N, c_rope]
        x_i_passthrough = x_i[:, :, c_rope:]  # untouched dims
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x_prime[i, seq_len_prime:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()


def rope_apply_prime_qk(q, k, grid_sizes, grid_sizes_prime, freqs, offsets=None):
    q = rope_apply_prime(q, grid_sizes, grid_sizes_prime, freqs, offsets=offsets)
    k = rope_apply_prime(k, grid_sizes, grid_sizes_prime, freqs, offsets=offsets)
    return q, k


class ChannelLastConv1d(nn.Conv1d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ConvMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim,
                                    dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w3 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 support_x_prime=False,
                 x_rope_offsets=None,
                 x_prime_rope_offsets=None
                 ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.support_x_prime = support_x_prime

        self.x_rope_offsets = x_rope_offsets
        self.x_prime_rope_offsets = x_prime_rope_offsets

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        
        # x' specific layers
        if support_x_prime:
            self.q_prime = nn.Linear(dim, dim)
            self.k_prime = nn.Linear(dim, dim)
            self.v_prime = nn.Linear(dim, dim)
            self.o_prime = nn.Linear(dim, dim)
            self.norm_q_prime = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
            self.norm_k_prime = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def asymmetric_attention_reverse(self, q, k, v, q_prime, k_prime, v_prime, 
                        seq_lens, grid_sizes, freqs, dtype, 
                        window_size=(-1, -1), dropout_p=0.,
                        seq_lens_prime=None, grid_sizes_prime=None,
                        x_rope_offsets=None, x_prime_rope_offsets=None):
        """
        Asymmetric attention implementation
        - Normal mode:
        - x' cannot see x (q_prime of x' cannot attend to k of x)
        - x can see x' (q of x can attend to k_prime of x')
        
        Args:
            q, k, v: query, key, value from x (video)
            q_prime, k_prime, v_prime: query, key, value from x'
            seq_lens: sequence lengths of x
            grid_sizes: grid sizes of x (F, H, W)
            freqs: RoPE frequencies
            dtype: data type
            seq_lens_prime: sequence length of x', if provided apply special position encoding
            grid_sizes_prime: grid sizes of x', if provided apply special position encoding

        Return:
            out_x: output of x
            out_x_prime: output of x'
        """
        b, s_x, n, d = q.shape
        _, s_x_prime, _, _ = q_prime.shape
        
        q, k = rope_apply_qk(q, k, grid_sizes, freqs, offsets=x_rope_offsets)

        q_prime, k_prime = rope_apply_prime_qk(q_prime, k_prime, grid_sizes, grid_sizes_prime, freqs, offsets=x_prime_rope_offsets)

        k_combined = torch.cat([k, k_prime], dim=1)
        v_combined = torch.cat([v, v_prime], dim=1)
        
        out_x = attention(
            q.to(dtype),
            k_combined.to(dtype),
            v=v_combined.to(dtype),
            k_lens=None,
            window_size=window_size,
            dropout_p=dropout_p
        )

        out_x_prime = attention(
            q_prime.to(dtype), 
            k_prime.to(dtype), 
            v=v_prime.to(dtype),
            k_lens=None,
            window_size=window_size,
            dropout_p=dropout_p
        )

        return out_x.to(dtype), out_x_prime.to(dtype)

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype, t, x_prime=None, seq_lens_prime=None, grid_sizes_prime=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            x_prime(Tensor, optional): Shape [B, L', C]
            seq_lens_prime(Tensor, optional): Shape [B]
            grid_sizes_prime(Tensor, optional): Shape [B, 3], the second dimension contains (F, H, W)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if self.support_x_prime and x_prime is not None:
            x_ndim = grid_sizes.shape[-1]
            if x_ndim == 3:
                b_prime, s_prime = x_prime.shape[:2]
                def qkv_fn_prime(x_prime):
                    q_prime = self.norm_q_prime(self.q_prime(x_prime.to(dtype))).view(b_prime, s_prime, n, d)
                    k_prime = self.norm_k_prime(self.k_prime(x_prime.to(dtype))).view(b_prime, s_prime, n, d)
                    v_prime = self.v_prime(x_prime.to(dtype)).view(b_prime, s_prime, n, d)
                    return q_prime, k_prime, v_prime
            else:
                b_prime, s_prime = x_prime.shape[:2]
                def qkv_fn_prime(x_prime):
                    q_prime = self.norm_q(self.q(x_prime.to(dtype))).view(b_prime, s_prime, n, d)
                    k_prime = self.norm_k(self.k(x_prime.to(dtype))).view(b_prime, s_prime, n, d)
                    v_prime = self.v(x_prime.to(dtype)).view(b_prime, s_prime, n, d)
                    return q_prime, k_prime, v_prime

            q_prime, k_prime, v_prime = qkv_fn_prime(x_prime)
            
            out_x, out_x_prime = self.asymmetric_attention_reverse(
                q, k, v, q_prime, k_prime, v_prime,
                seq_lens, grid_sizes, freqs, dtype, 
                window_size=self.window_size,
                seq_lens_prime=seq_lens_prime,
                grid_sizes_prime=grid_sizes_prime,
                x_rope_offsets=self.x_rope_offsets,
                x_prime_rope_offsets=self.x_prime_rope_offsets
            )
            
            # output
            out_x = out_x.flatten(2)
            out_x = self.o(out_x)

            if x_ndim == 3:
                out_x_prime = out_x_prime.flatten(2)
                out_x_prime = self.o_prime(out_x_prime)
            else:
                out_x_prime = out_x_prime.flatten(2)
                out_x_prime = self.o(out_x_prime)
            
            return out_x, out_x_prime
        else:
            q, k = rope_apply_qk(q, k, grid_sizes, freqs)

            x = attention(
                q.to(dtype), 
                k.to(dtype), 
                v=v.to(dtype),
                k_lens=seq_lens,
                window_size=self.window_size)
            x = x.to(dtype)

            # output
            x = x.flatten(2)
            x = self.o(x)
            return x


class WanT2VCrossAttention(WanSelfAttention):
    def qkv_fn(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        return q, k, v

    def forward(self, x, context, context_lens, dtype, t):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v = self.qkv_fn(x, context)

        # compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 additional_emb_length=8):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.additional_emb_length = additional_emb_length

    def qkv_fn(self, x, context):
        context_img = context[:, : self.additional_emb_length]
        context = context[:, self.additional_emb_length :]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        return q, k, v, k_img, v_img
    
    def forward(self, x, context, context_lens, dtype, t):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v, k_img, v_img = self.qkv_fn(x, context)

        img_x = attention(
            q.to(dtype), 
            k_img.to(dtype), 
            v_img.to(dtype), 
            k_lens=None
        )
        img_x = img_x.to(dtype)
        # compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens, dtype, t):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        # compute attention
        x = attention(q, k, v, k_lens=context_lens)
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    'cross_attn': WanCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 support_x_prime=False,
                 x_rope_offsets=None,
                 x_prime_rope_offsets=None
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.support_x_prime = support_x_prime

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, support_x_prime=support_x_prime, 
                                          x_rope_offsets=x_rope_offsets, x_prime_rope_offsets=x_prime_rope_offsets)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        
        # x' specific layers
        if support_x_prime:
            self.norm1_prime = WanLayerNorm(dim, eps)
            self.norm2_prime = WanLayerNorm(dim, eps)
            self.norm3_prime = WanLayerNorm(
                dim, eps,
                elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.cross_attn_prime = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                                num_heads,
                                                                                (-1, -1),
                                                                                qk_norm,
                                                                                eps)
            self.ffn_prime = nn.Sequential(
                nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, dim))
            
            self.modulation_prime = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.float32,
        t=0,
        x_prime=None,
        seq_lens_prime=None,
        grid_sizes_prime=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] - 视频 tokens
            e(Tensor): Shape [B, 6, C] - modulation for x
            seq_lens(Tensor): Shape [B], length of each video sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W) - 视频网格大小
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            x_prime(Tensor, optional): Shape [B, L', C] - 图像 tokens
            seq_lens_prime(Tensor, optional): Shape [B], length of each image sequence in batch
            grid_sizes_prime(Tensor, optional): Shape [B, 3], the second dimension contains (F, H, W) - 图像网格大小
        """
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        if self.support_x_prime and x_prime is not None:
            e_prime = [ item[:,:x_prime.shape[1],:] for item in e]
            temp_x_prime = self.norm1_prime(x_prime) * (1 + e_prime[1]) + e_prime[0]
            temp_x_prime = temp_x_prime.to(dtype)
            
            y, y_prime = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype, t=t, x_prime=temp_x_prime, seq_lens_prime=seq_lens_prime, grid_sizes_prime=grid_sizes_prime)
            x = x + y * e[2]
            x_prime = x_prime + y_prime * e_prime[2]
            
            def cross_attn_ffn(x, context, context_lens, e, norm2, norm3, cross_attn, ffn):
                # cross-attention
                x = x + cross_attn(norm3(x), context, context_lens, dtype, t=t)

                # ffn function
                temp_x = norm2(x) * (1 + e[4]) + e[3]
                temp_x = temp_x.to(dtype)
                
                y = ffn(temp_x)
                x = x + y * e[5]
                return x

            x = cross_attn_ffn(x, context, context_lens, e, self.norm2, self.norm3, self.cross_attn, self.ffn)
            x_prime = cross_attn_ffn(x_prime, context, context_lens, e_prime, 
                                   self.norm2_prime, self.norm3_prime, self.cross_attn_prime, self.ffn_prime)
            
            return x, x_prime
        else:
            y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype, t=t)
            x = x + y * e[2]

            def cross_attn_ffn(x, context, context_lens, e):
                # cross-attention
                x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype, t=t)

                # ffn function
                temp_x = self.norm2(x) * (1 + e[4]) + e[3]
                temp_x = temp_x.to(dtype)
                
                y = self.ffn(temp_x)
                x = x + y * e[5]
                return x

            x = cross_attn_ffn(x, context, context_lens, e)
            return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        if e.dim() > 2:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens



class WanTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        cross_attn_type=None,
        support_x_prime=False,
        temporal_rope_scaling_factor=1.0,
        x_rope_offsets=None,
        x_prime_rope_offsets=None
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 't2a', 'tf2v']
        self.model_type = model_type

        is_audio_type = "a" in self.model_type
        is_video_type = "v" in self.model_type
        assert is_audio_type ^ is_video_type, "Either audio or video model should be specified"
        if is_audio_type:
            assert len(patch_size) == 1 and patch_size[0] == 1, "Audio model should only accept 1 dimensional input, and we dont do patchify"

        self.is_audio_type = is_audio_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.support_x_prime = support_x_prime
        self.temporal_rope_scaling_factor = temporal_rope_scaling_factor

        # embeddings
        if self.is_audio_type:
            self.patch_embedding = nn.Sequential(
                ChannelLastConv1d(in_dim, dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(dim, dim * 4, kernel_size=7, padding=3),
            )
        else:
            self.patch_embedding = nn.Conv3d(
                in_dim, dim, kernel_size=patch_size, stride=patch_size)
            
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        if cross_attn_type is None:
            cross_attn_type = 't2v_cross_attn' if model_type in ['t2v', 't2a', 'ti2v', 'tf2v'] else 'i2v_cross_attn'

        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, 
                              support_x_prime=support_x_prime, 
                              x_rope_offsets=x_rope_offsets, x_prime_rope_offsets=x_prime_rope_offsets)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        self.set_rope_params()

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        elif model_type == 'tf2v':
            self.img_emb = MLPProj(4096, dim)
        
        self.teacache = None
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None
        self.gradient_checkpointing = False
        self.sp_world_size = 1
        self.sp_world_rank = 0

        self.init_weights()

    def set_rope_params(self):
        dim = self.dim
        num_heads = self.num_heads
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        if self.is_audio_type:
            self.freqs = rope_params(1024, d - 4 * (d // 6), freqs_scaling=self.temporal_rope_scaling_factor)
        else:
            self.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
                                dim=1)

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")

    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True,
    ):
        self.teacache = TeaCache(
            coefficients, num_steps, rel_l1_thresh=rel_l1_thresh, num_skip_start_steps=num_skip_start_steps, offload=offload
        )

    def share_teacache(
        self,
        transformer = None,
    ):
        self.teacache = transformer.teacache

    def disable_teacache(self):
        self.teacache = None

    def enable_cfg_skip(self, cfg_skip_ratio, num_steps):
        if cfg_skip_ratio != 0:
            self.cfg_skip_ratio = cfg_skip_ratio
            self.current_steps = 0
            self.num_inference_steps = num_steps
        else:
            self.cfg_skip_ratio = None
            self.current_steps = 0
            self.num_inference_steps = None

    def share_cfg_skip(
        self,
        transformer = None,
    ):
        self.cfg_skip_ratio = transformer.cfg_skip_ratio
        self.current_steps = transformer.current_steps
        self.num_inference_steps = transformer.num_inference_steps

    def disable_cfg_skip(self):
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None

    def enable_riflex(
        self,
        k = 6,
        L_test = 66,
        L_test_scale = 4.886,
    ):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                get_1d_rotary_pos_embed_riflex(1024, self.d - 4 * (self.d // 6), use_real=False, k=k, L_test=L_test, L_test_scale=L_test_scale),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def disable_riflex(self):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                rope_params(1024, self.d - 4 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def enable_multi_gpus_inference(self,):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn)
            
    def prepare_transformer_block_kwargs(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        first_frame_is_clean=False,
        id_images=None,
        id_seq_len=None,
        spkid=None,
        spkid_seq_len=None
    ):
        device = next(self.patch_embedding.parameters()).device
            
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        if self.is_audio_type:
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[1:2], dtype=torch.long) for u in x])
            if spkid is not None:
                x_prime = [self.patch_embedding(u.unsqueeze(0)) for u in spkid]
                grid_sizes_prime = torch.stack(
                    [torch.tensor(u.shape[1:2], dtype=torch.long) for u in x_prime]
                )   
                seq_lens_prime = torch.tensor([u.size(1) for u in x_prime], dtype=torch.long)

                if self.sp_world_size > 1:
                    spkid_seq_len = int(math.ceil(spkid_seq_len / self.sp_world_size)) * self.sp_world_size
                assert seq_lens_prime.max() <= spkid_seq_len, f"Sequence length {seq_lens_prime.max()} exceeds maximum {spkid_seq_len}."
                x_prime = torch.cat([
                      torch.cat([u, u.new_zeros(1, spkid_seq_len - u.size(1), u.size(2))],
                                dim=1) for u in x_prime
                ])
                if x_prime.shape[0] > 1:
                    spkid_seq_len *= x_prime.shape[0]
                    x_prime = x_prime.flatten(0, 1).unsqueeze(0)
        else:
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]

            if id_images is not None:
                id_images_frames = id_images.size(2)
                x_prime = [self.patch_embedding(u.unsqueeze(0)) for u in id_images]
                grid_sizes_prime = torch.stack(
                    [torch.tensor(u.shape[2:], dtype=torch.long) for u in x_prime])
                x_prime = [u.flatten(2).transpose(1, 2) for u in x_prime]
                seq_lens_prime = torch.tensor([u.size(1) for u in x_prime], dtype=torch.long)
                
                if self.sp_world_size > 1:
                    id_seq_len = int(math.ceil(id_seq_len / self.sp_world_size)) * self.sp_world_size
                assert seq_lens_prime.max() <= id_seq_len, f"ID sequence length {seq_lens_prime.max()} exceeds maximum {id_seq_len}."
                x_prime = torch.cat([
                    torch.cat([u, u.new_zeros(1, id_seq_len - u.size(1), u.size(2))],
                            dim=1) for u in x_prime
                ])

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len, f"Sequence length {seq_lens.max()} exceeds maximum {seq_len}."
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        if t.dim() == 1:
            if first_frame_is_clean:
                t = torch.ones((t.size(0), seq_len), device=t.device, dtype=t.dtype) * t.unsqueeze(1)
                _first_images_seq_len = grid_sizes[:, 1:].prod(-1)
                for i in range(t.size(0)):
                    t[i, :_first_images_seq_len[i]] = 0
            else:
                if spkid_seq_len is not None:
                    t_id = t.unsqueeze(1).expand(t.size(0), spkid_seq_len)
                    t_id = t_id.flatten()
                    t_id = torch.zeros_like(t_id)
                t = t.unsqueeze(1).expand(t.size(0), seq_len)
            
        with amp.autocast(dtype=torch.bfloat16):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

            if spkid is not None:
                e_prime = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t_id).unflatten(0, (bt, spkid_seq_len)).float())
                e0_prime = self.time_projection(e_prime).unflatten(2, (6, self.dim))

            if id_images is not None:
                t_id = torch.zeros_like(t)[:bt * id_seq_len]
                e_prime = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t_id).unflatten(0, (bt, id_seq_len)).float())
                e0_prime = self.time_projection(e_prime).unflatten(2, (6, self.dim))

        # Context Parallel
        if self.sp_world_size > 1:
            current_len = x.shape[1]
            pad_size = (-current_len ) % self.sp_world_size  

            if pad_size > 0:
                padding = torch.zeros(
                    x.shape[0], pad_size, x.shape[2],
                    device=x.device,
                    dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=1)
                e_padding = torch.zeros(
                    e.shape[0], pad_size, e.shape[2],
                    device=e.device,
                    dtype=e.dtype
                )
                e = torch.cat([e, e_padding], dim=1)
                e0_padding = torch.zeros(
                    e0.shape[0], pad_size, e0.shape[2], e0.shape[3],
                    device=e0.device,
                    dtype=e0.dtype
                )
                e0 = torch.cat([e0, e0_padding], dim=1)

            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            
            e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
            e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        if id_images is not None or spkid is not None:
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                x_prime=x_prime,
                e_prime=e0_prime,
                seq_lens_prime=seq_lens_prime,
                grid_sizes_prime=grid_sizes_prime)
        else:
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                x_prime=None,
                e_prime=None,
                seq_lens_prime=None,
                grid_sizes_prime=None)

        return x, e, kwargs
        
    def post_transformer_block_out(
        self, 
        x, 
        grid_sizes, 
        e,
        cond_flag=True
    ):
        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # unpatchify
        if self.is_audio_type:
            grid_sizes = [gs[0] for gs in grid_sizes]
            assert len(x) == len(grid_sizes)
            x = [u[:gs] for u, gs in zip(x, grid_sizes)]
        else:
            x = self.unpatchify(x, grid_sizes)

        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()

        return x

    @cfg_skip()
    def forward(
        self,
        x=None,
        t=None,
        context=None,
        seq_len=None,
        clip_fea=None,
        y=None,
        cond_flag=True,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor], *optional*):
                List of input video tensors, each with shape [C_in, F, H, W].
                Can be None in audio-only mode.
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            List[Tensor] or Tensor:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        dtype = x.dtype

        x, e, kwargs = self.prepare_transformer_block_kwargs(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y,
            first_frame_is_clean=first_frame_is_clean,
        )
    
        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                modulated_inp = kwargs['e']
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc
    
        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)
                            return custom_forward
                            
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            kwargs['e'],
                            kwargs['seq_lens'],
                            kwargs['grid_sizes'],
                            kwargs['freqs'],
                            kwargs['context'],
                            kwargs['context_lens'],
                            dtype,
                            t,
                            **ckpt_kwargs,
                        )
                    else:
                        x = block(
                            x,
                            kwargs['e'],
                            kwargs['seq_lens'],
                            kwargs['grid_sizes'],
                            kwargs['freqs'],
                            kwargs['context'],
                            kwargs['context_lens'],
                            dtype, 
                            t, 
                        )
                        
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        kwargs['e'],
                        kwargs['seq_lens'],
                        kwargs['grid_sizes'],
                        kwargs['freqs'],
                        kwargs['context'],
                        kwargs['context_lens'],
                        dtype,
                        t,
                        **ckpt_kwargs
                    )
                else:
                    x = block(
                        x,
                        kwargs['e'],
                        kwargs['seq_lens'],
                        kwargs['grid_sizes'],
                        kwargs['freqs'],
                        kwargs['context'],
                        kwargs['context_lens'],
                        dtype, 
                        t, 
                    )

        x = self.post_transformer_block_out(
            x, 
            kwargs['grid_sizes'],
            e, 
            cond_flag
        )
        
        return x


    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if isinstance(self.patch_embedding, nn.Sequential):
            for sub_module in self.patch_embedding:
                if hasattr(sub_module, 'weight'):
                    nn.init.xavier_uniform_(sub_module.weight.flatten(1))
                    return
            raise ValueError("No sub-module with 'weight' found in Sequential")
        else:
            nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
    

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                from diffusers.models.modeling_utils import \
                    load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(state_dict)
                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )
        
        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
            model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight']
            model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
            state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']
        
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        model = model.to(torch_dtype)
        return model