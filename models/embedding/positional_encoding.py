import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_embed, max_len=256, device = torch.device('cpu')):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        # encoding이 학습되지 않도록 false를 부여해야 한다. 
        # 왜냐하면 position_encoding은 학습되는 parameter가 아니기 때문이다!
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2)) * -(math.log(10000.0) / d_embed)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)
        
    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out