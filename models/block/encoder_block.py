import os
import copy
import torch.nn as nn
from models.layer.residual_connection_layer import ResidualConnectionLayer

class EncoderBlock(nn.Module):
    # 인코더 블록은 크게 두가지로 나뉨
    # attention layer와 position feed forward layer로 구성됨
    def __init__(self, self_attention, position_ff, norm, dr_rate=0):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        #jieun_add
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        
        self.position_ff = position_ff
        # self.residuals = [ResidualConnectionLayer() for _ in range(2)] # type: ignore
        #jieun_add
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        
    def forward(self, src, src_mask):
        # 첫번째 input은 전체의 input인 x가 되고
        # 가장 마지막 output은 context를 return 해야한다. 
        out = src
        # residuals[0]은 multi-head-attention-layer를 감싸고, 
        # residuals[1]은 position-feed-forward-layer를 감싼다! 
        
        # out = self.residuals[0](out, lambda out : self.self_attention(query = out, key = out, value = out, mask = src_mask))
        # out = self.residuals[1](out, self.position_ff) 
        
        #jieun_add
        out = self.residual1(out, lambda out : self.self_attention(query = out, key = out, value = out, mask = src_mask))
        out = self.residual2(out, self.position_ff)
        
        return out #최종 인자가 x, mask가 아니라 x(q),x(k),x(v),mask가 된다.
    