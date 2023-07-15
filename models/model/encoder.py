import copy
import torch.nn as nn

class Encoder(nn.Module):
    # n_layer: encoder blook 수
    def __init__(self, encoder_block, n_layer, norm): 
        super(Encoder, self).__init__()
        # self.layers = []
        # for i in range(n_layer):
        #     self.layers.append(copy.deepcopy(encoder_block))
        # jieun_remove
        self.n_layer = n_layer
        self.norm = norm
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(self.n_layer)])
        
        
    def forward(self,src, src_mask): 
        # mask 인자를 받고 각 sublayer의 forward()로 넘겨주기 위해 수정해야 함
        # 최종 output은 x, mask 이다! 
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        
        #jieun_add
        out = self.norm(out)
        
        return out