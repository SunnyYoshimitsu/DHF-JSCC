# Auto-generated ops module
import torch
import torch.nn as nn
from compressai.ops import LowerBound
from compressai.layers import GDN

class Low_bound(torch.autograd.Function):
    """Lower bound operation"""
    @staticmethod
    def forward(ctx, x, bound):
        return LowerBound.apply(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# Export GDN from CompressAI
GDN = GDN
