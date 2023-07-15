import torch.nn as nn
import torch.nn.functional as F


# position-wise feed forward layer
# 단순하게 2개의 fc layer를 가지고 있다! 
# ff layer는 multi-head attention layer의 output을 input으로 받아 연산을 수행!
# 그 다음 encoder block에게 output을 넘겨준다! 
# 논문에서는 Fc layer의 output에 RELU()를 적용함!

#Residual connection layer을 이용해서 back propagation 도중 발생할 수 있는
#gradient vanishing을 방지하자! 

class ResidualConnectionLayer(nn.Module):
    
    def __init__(self,norm, dr_rate=0):
        super(ResidualConnectionLayer, self).__init__()
        #jieun_add
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)
        
    def forward(self, x, sub_layer):
        out = x 
        out = self.norm(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x
        
        return out
                                   