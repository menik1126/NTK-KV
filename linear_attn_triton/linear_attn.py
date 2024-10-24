# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

import functools
from typing import Optional, Tuple

from packaging import version
import torch
import triton
import triton.language as tl


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


@torch.jit.script
def normalize_output(q, k, o):
    k = k.cumsum(-2)
    z = (q * k).sum(-1, keepdim=True)
    return o / (z + 1e-10)


def flash_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    phi_k_v_sum: torch.Tensor = None,
    phi_k_sum: torch.Tensor = None,
):
    o, final_state = FusedRecurrentLinearAttentionFunction.apply(q, k, v, 1, phi_k_v_sum, True)

    k = (k.cumsum(-2) + phi_k_sum) if phi_k_sum is not None else k.cumsum(-2)
    d = (q * k).sum(-1, keepdim=True)

    o = o / (d + 1e-10)

    return o, (final_state, k[..., -1:, :])


@triton.jit
def fused_recurrent_linear_attn_fwd_kernel(
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V]
    o,  # output [B, H, L, V]
    h0,
    ht,  # final hidden state [B, H, K, V]

    s_qk_h,  # stride size: L * K
    s_vo_h,  # stride size: L * V

    scale,
    B,  # batch size
    H,  # H
    T,  # T
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale

        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += K
        p_k += K
        p_o += V
        p_v += V

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_recurrent_linear_attn_bwd_kernel(
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V]

    do,  # gradient of output [B, H, L, V]
    dq,  # gradient of query [NV, B, H, L, K]
    dk,  # gradient of key [NV, B, H, L, K]
    dv,  # gradient of value [NK, B, H, L, V]
    h0,  # initial hidden state initialization [B, H, K, V]

    s_qk_h,  # stride size: L * K
    s_vo_h,  # stride size: L * V
    scale,  # K ** -0.5

    B,  # B
    H,  # H
    T,  # T
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)

    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

        b_h += b_k[:, None] * b_v[None, :]
        _d_q = b_h * b_do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += K
        p_do += V
        p_v += V
        p_dq += K

    # sync threads
    tl.debug_barrier()

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    d_h = tl.zeros([BK, BV], dtype=tl.float32)

    for _ in range(T):
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        d_h += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(d_h * b_v[None, :], axis=1)
        d_v = tl.sum(d_h * b_k[:, None], axis=0)

        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_do -= V
        p_q -= K
        p_k -= K
        p_v -= V
        p_dk -= K
        p_dv -= V


class FusedRecurrentLinearAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, scale, initial_state=None, output_final_state=False):
        B, H, T, K = q.shape
        V = v.shape[-1]

        BK, BV = min(K, 32), min(V, 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_warps = 1
        num_stages = 1

        o = q.new_empty(NK, B, H, T, V)
        final_state = q.new_empty(B, H, K, V) if output_final_state else None

        grid = (NV, NK, B * H)
        fused_recurrent_linear_attn_fwd_kernel[grid](
            q, k, v, o, initial_state, final_state,
            q.stride(1),
            v.stride(1), scale,
            B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            num_warps=num_warps,
            num_stages=num_stages
        )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, initial_state)
        ctx.scale = scale
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht=None):
        q, k, v, initial_state = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        scale = ctx.scale

        BK, BV = min(K, 32), min(V, 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_warps = 1
        num_stages = 1

        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)
        grid = (NV, NK, B * H)

        fused_recurrent_linear_attn_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv, initial_state,
            q.stride(1),
            v.stride(1),
            scale,
            B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq, dk, dv, None, None, None


def fused_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, final_state = FusedRecurrentLinearAttentionFunction.apply(q, k, v, scale, initial_state, output_final_state)
    if normalize:
        o = normalize_output(q * scale, k, o)
    return o, final_state


def fused_recurrent_linear_attn_w_scale(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    phi_k_v_sum: torch.Tensor = None,
    phi_k_sum: torch.Tensor = None,
):
    o, final_state = FusedRecurrentLinearAttentionFunction.apply(q, k, v, 1, phi_k_v_sum, True)

    k = k.cumsum(-2) + phi_k_sum
    d = torch.matmul(q, k).sum(-1, keepdim=True)
    o = o / d

    return o, (final_state, k[..., -1:, :])


if version.parse(torch.__version__) >= version.parse("2.4"):
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")
else:
    autocast_custom_fwd = torch.cuda.amp.custom_fwd
    autocast_custom_bwd = torch.cuda.amp.custom_bwd


@triton.jit
def fused_chunk_linear_attn_fwd_kernel(
    q,  # query [B, H, T, K]
    k,  # key [B, H, T, V]
    v,  # value [B, H, T, V]
    o,  # output [B, H, T, V]
    h0,
    ht,
    s_qk_h,  # stride size: T * K
    s_qk_t,  # stride size: K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: T * V
    s_vo_t,  # stride size: V
    s_vo_d,  # stride size: 1
    # scale,
    B,  # batch size
    H,  # H
    T,  # T
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)

    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]
    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    # make block pointers
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh+i_k*B*H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # [BT, BT]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        # [BT, BV]
        b_o = tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        if CHECK and i == 0:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def fused_chunk_linear_attn_bwd_kernel(
    q,  # query [B, H, T, K]
    k,  # key [B, H, T, V]
    v,  # value [B, H, T, V]
    do,  # gradient of output [B, H, T, V]
    dq,  # gradient of query [NV, B, H, T, K]
    dk,  # gradient of key [NV, B, H, T, K]
    dv,  # gradient of value [NK, B, H, T, V]

    h0,  # initial state of the chunk [B, H, K, V]

    s_qk_h,  # stride size: T * K
    s_qk_t,  # stride size: K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: T * V
    s_vo_t,  # stride size: V
    s_vo_d,  # stride size: 1
    scale,  # K ** -0.5
    B,  # B
    H,  # H
    T,  # T
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)

    m_s = o_i[:, None] >= o_i[None, :]
    # [BV, BK]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v*B*H) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i*BT, i_k*BK), (BT, BK), (1, 0))

        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [V, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, V]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # [BT, BT]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        # [BT, BK]
        b_dq = tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)
        # [BV, BK]
        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # sync threads
    b_h = None
    tl.debug_barrier()
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    m_s = o_i[:, None] <= o_i[None, :]
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh+i_v*B*H) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (T - i*BT, i_k*BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh+i_k*B*H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i*BT, i_v*BV), (BT, BV), (1, 0))
        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # [BT, BT]
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0).to(b_q.dtype)
        # [BT, BT]
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0).to(b_q.dtype)
        # [BT, BK]
        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        # [BT, BV]
        b_dv = tl.dot(b_s, b_do, allow_tf32=False)
        if CHECK and i == 1:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)
        else:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class FusedChunkLinearAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale, initial_state, output_final_state):
        B, H, T, K, V = *k.shape, v.shape[-1]
        BT = 64
        BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_warps = 4
        num_stages = 1

        o = q.new_empty(NK, B, H, T, V)
        final_state = q.new_empty(B, H, K, V, dtype=torch.float) if output_final_state else None
        # the bug still exists even for Triton 2.2 on H100 GPUs
        # so we always enable initial checks
        CHECK = True
        if version.parse(triton.__version__) < version.parse('2.2.0'):
            import warnings
            warnings.warn(
                "Triton<2.2.0 detected for running this kernel, "
                "which is known to have some weird compiler issues (refer to https://github.com/openai/triton/issues/2852) "
                "that lead to significant precision loss. "
                "We've add some initial condition checks to resolve this, sadly at the sacrifice of the speed. "
                "For optimal performance, it is recommended to install Triton>=2.2.0 (if possible)."
            )
            CHECK = True

        grid = (NV, NK, B * H)
        fused_chunk_linear_attn_fwd_kernel[grid](
            q, k, v, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            scale,
            B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            CHECK=CHECK,
            num_warps=num_warps,
            num_stages=num_stages
        )
        o = o.sum(0) if NK > 1 else o[0]

        ctx.save_for_backward(q, k, v, initial_state)
        ctx.scale = scale
        ctx.CHECK = CHECK
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        q, k, v, initial_state = ctx.saved_tensors
        B, H, T, K, V = *k.shape, v.shape[-1]
        scale = ctx.scale

        BT = 64
        BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_warps = 4
        num_stages = 1

        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)
        grid = (NV, NK, B * H)

        fused_chunk_linear_attn_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv, initial_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            scale,
            B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            CHECK=ctx.CHECK,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def fused_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `(B, H, T, K)`
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        scale (Optional[int]):
            Scale factor for linear attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
        normalize (bool):
            Whether to normalize the output. Default: `True`.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, final_state = FusedChunkLinearAttentionFunction.apply(q, k, v, scale, initial_state, output_final_state)
    if normalize:
        o = normalize_output(q * scale, k, o)
    return o, final_state


@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    k,
    v,
    h,
    h0,
    ht,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BV]
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q,
    k,
    v,
    h,
    o,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
    b_s = tl.where(m_s, b_s, 0)

    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_linear_attn_bwd_kernel_dh(
    q,
    do,
    dh,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, V]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BK, BV]
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False)


@triton.jit
def chunk_linear_attn_bwd_kernel_dqkv(
    q,
    k,
    v,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
    b_s = tl.where(o_i[:, None] <= o_i[None, :], b_s, 0)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (NT * K, V), (s_h_t, 1), (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh)*s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h, allow_tf32=False) * scale
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) + tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    # [BT, BT]
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale, 0).to(b_q.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk += tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))

    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


class ChunkLinearAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale, initial_state, output_final_state):
        B, H, T, K, V = *q.shape, v.shape[-1]
        BT = 64
        BK, BV = min(64, triton.next_power_of_2(K)), min(64, triton.next_power_of_2(V))
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        ctx.scale = scale

        final_state = None
        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float32, requires_grad=False)

        h = q.new_empty(B, H, NT * K, V)
        grid = (NK, NV, B * H)
        chunk_linear_attn_fwd_kernel_h[grid](
            k, v, h, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            num_warps=num_warps,
            num_stages=num_stages
        )
        grid = (NV, NT, B * H)
        o = torch.empty_like(v)
        chunk_linear_attn_fwd_kernel_o[grid](
            q, k, v, h, o,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            scale,
            T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ctx.save_for_backward(q, k, v, h)
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        q, k, v, h = ctx.saved_tensors

        B, H, T, K, V = *q.shape, v.shape[-1]
        BT = 64
        BK, BV = min(64, triton.next_power_of_2(K)), min(32 if q.dtype == torch.float32 else 64, triton.next_power_of_2(V))
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = ctx.scale

        dh = q.new_empty(B, H, NT * K, V)
        grid = (NK, NV, B * H)
        chunk_linear_attn_bwd_kernel_dh[grid](
            q, do, dh,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            dh.stride(1), dh.stride(2),
            scale,
            T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )

        grid = (NK, NT, B * H)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = v.new_empty(NK, *v.shape)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        chunk_linear_attn_bwd_kernel_dqkv[grid](
            q, k, v, h, do, dh, dq, dk, dv,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            dh.stride(1), dh.stride(2),
            scale,
            T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dv = dv.sum(0)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `(B, H, T, K)`
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        scale (Optional[int]):
            Scale factor for the linear attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
        normalize (bool):
            Whether to normalize the output. Default: `True`.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, final_state = ChunkLinearAttentionFunction.apply(q, k, v, scale, initial_state, output_final_state)
    if normalize:
        o = normalize_output(q * scale, k, o)
    return o, final_state
