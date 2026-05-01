import torch
import math

# Ignoring because I'm on Mac and VSCode is annoying
import triton # type: ignore
import triton.language as tl  # type: ignore

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):

    batch_index = tl.program_id(1)
    query_tile_index = tl.program_id(0)

    Q_offset = batch_index * stride_qb
    K_offset = batch_index * stride_kb
    V_offset = batch_index * stride_vb
    O_offset = batch_index * stride_ob
    L_offset = batch_index * stride_lb

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + Q_offset,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + O_offset,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + K_offset,
        shape=(D, N_KEYS), 
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + V_offset,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_tile = tl.load(Q_block_ptr)

    m_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    for k_tile_index in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr)
        V_tile = tl.load(V_block_ptr)

        S_ij = tl.dot(Q_tile, K_tile) * scale

        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = k_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            S_ij = tl.where(causal_mask, S_ij, float('-inf'))

        m_ij = tl.max(S_ij, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        P_tilde_ij = tl.exp(S_ij - m_i_new[:, None])

        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(P_tilde_ij, axis=1)
        
        P_tilde_ij_cast = P_tilde_ij.to(V_tile.dtype)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(P_tilde_ij_cast, V_tile, acc=acc)

        m_i = m_i_new
        l_i = l_i_new

        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_tile = acc / l_i[:, None]
    
    tl.store(O_block_ptr, O_tile.to(O_ptr.dtype.element_ty))

    L_i = m_i + tl.log(l_i)
    
    l_ptrs = L_ptr + L_offset + query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    tl.store(l_ptrs, L_i)


@torch.compile
def flash_backward_fn(Q, K, V, O, dO, L, is_causal=False):
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    
    # S = QK^T / sqrt(d)
    S = torch.bmm(Q, K.transpose(1, 2)) * scale
    
    if is_causal:
        q_idx = torch.arange(Q.shape[1], device=Q.device)[:, None]
        k_idx = torch.arange(K.shape[1], device=K.device)[None, :]
        mask = q_idx >= k_idx
        S = torch.where(mask[None, :, :], S, float('-inf'))
        
    # P = exp(S - L)
    P = torch.exp(S - L.unsqueeze(-1))
    
    # D = rowsum(O * dO)
    D = torch.sum(O * dO, dim=-1, keepdim=True)
    
    # dV = P^T dO
    dV = torch.bmm(P.transpose(1, 2), dO)
    
    # dP = dO V^T
    dP = torch.bmm(dO, V.transpose(1, 2))
    
    # dS = P * (dP - D)
    dS = P * (dP - D)
    
    # dQ = dS K / sqrt(d)
    dQ = torch.bmm(dS, K) * scale

    # dK = dS^T Q / sqrt(d)
    dK = torch.bmm(dS.transpose(1, 2), Q) * scale
    
    return dQ, dK, dV


class TritonFlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_q, d = Q.shape
        _, N_k, _ = K.shape

        O = torch.empty_like(Q)
        L = torch.empty((B, N_q), device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        grid = (triton.cdiv(N_q, Q_TILE_SIZE), B)
        scale = 1.0 / (d ** 0.5)

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N_q, N_KEYS=N_k,
            scale=scale,
            is_causal=is_causal,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_backward_fn(Q, K, V, O, grad_out, L, is_causal)
        return dQ, dK, dV, None