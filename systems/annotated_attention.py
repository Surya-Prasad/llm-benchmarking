import math
import torch
import torch.cuda.nvtx as nvtx
from basics.basics.nn_utils import softmax 

def vanilla_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    attention_weights = softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    
    with nvtx.range("computing attention scores"):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        output = torch.matmul(attention_weights, V)

    return output