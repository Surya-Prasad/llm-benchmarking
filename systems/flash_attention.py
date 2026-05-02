import torch
import math

@torch.compile
def flash_backward_fn(Q, K, V, O, dO, L, is_causal=False):
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    
    S = torch.bmm(Q, K.transpose(1, 2)) * scale
    
    if is_causal:
        q_idx = torch.arange(Q.shape[1], device=Q.device)[:, None]
        k_idx = torch.arange(K.shape[1], device=K.device)[None, :]
        mask = q_idx >= k_idx
        S = torch.where(mask[None, :, :], S, float('-inf'))
        
    P = torch.exp(S - L.unsqueeze(-1))
    P = P.to(Q.dtype)
    D = torch.sum(O * dO, dim=-1, keepdim=True)
    dV = torch.bmm(P.transpose(1, 2), dO)
    dP = torch.bmm(dO, V.transpose(1, 2))
    dS = P * (dP - D)
    dS = (P * (dP - D)).to(Q.dtype)
    dQ = torch.bmm(dS, K) * scale
    dK = torch.bmm(dS.transpose(1, 2), Q) * scale
    
    return dQ, dK, dV

class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_q, d = Q.shape
        _, N_k, _ = K.shape

        B_q = 64
        B_k = 64

        O = torch.zeros_like(Q)
        L = torch.zeros(B, N_q, device=Q.device, dtype=Q.dtype)

        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)

        scale = 1.0 / math.sqrt(d)

        for i in range(T_q):
            q_start = i * B_q
            q_end = min(q_start + B_q, N_q)
            Qi = Q[:, q_start:q_end, :] 

            m_i = torch.full((B, q_end - q_start, 1), float('-inf'), device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros((B, q_end - q_start, 1), device=Q.device, dtype=Q.dtype)
            O_i = torch.zeros((B, q_end - q_start, d), device=Q.device, dtype=Q.dtype)

            for j in range(T_k):
                k_start = j * B_k
                k_end = min(k_start + B_k, N_k)
                Kj = K[:, k_start:k_end, :] 
                Vj = V[:, k_start:k_end, :] 

                S_ij = torch.bmm(Qi, Kj.transpose(1, 2)) * scale    

                m_i_new = torch.maximum(m_i, torch.max(S_ij, dim=-1, keepdim=True)[0])

                P_tilde_ij = torch.exp(S_ij - m_i_new)

                exp_diff = torch.exp(m_i - m_i_new)
                l_i_new = exp_diff * l_i + torch.sum(P_tilde_ij, dim=-1, keepdim=True)
                O_i_new = exp_diff * O_i + torch.bmm(P_tilde_ij, Vj)

                m_i = m_i_new
                l_i = l_i_new
                O_i = O_i_new

            O_i = O_i / l_i
            O[:, q_start:q_end, :] = O_i

            L_i = m_i.squeeze(-1) + torch.log(l_i.squeeze(-1))
            L[:, q_start:q_end] = L_i

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        dQ, dK, dV = flash_backward_fn(Q, K, V, O, grad_out, L, is_causal)
        
        return dQ, dK, dV, None