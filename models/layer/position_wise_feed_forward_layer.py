import torch.nn.functional as F
import torch.nn as nn

# position-wise feed forward layer
# 단순하게 2개의 fc layer를 가지고 있다! 
# ff layer는 multi-head attention layer의 output을 input으로 받아 연산을 수행!
# 그 다음 encoder block에게 output을 넘겨준다! 
# 논문에서는 Fc layer의 output에 RELU()를 적용함!

class PositionWiseFeedForwardLayer(nn.Module):
    
    def __init__(self, fc1, fc2,dr_rate=0.):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1 # d_embed, d_ff
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dr_rate)
        self.fc2 = fc2 # d_ff, d_model
        
    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    