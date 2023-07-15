import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F



class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, d_model, h, qkv_fc, out_fc,dr_rate=0.):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h=h
        self.q_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(out_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(out_fc) # (d_embed, d_model)
        self.out_fc = out_fc # (d_model, d_embed)
        #jieun_add
        self.dropout = nn.Dropout(dr_rate)
        
        #deepcopy를 호출한 이유는 서로 다른 weight를 갖고 별개로 운용되게 하기 위함이다.
        #out_fc 는 attention 계산 이후 거쳐가는 fc_layer
      
    # 잠깐 기본적인 q,k,v를 설명하자면,
    # query : 궁금한 token
    # key : 전체 문장들의 token
    # value : key와 동일한 token

    # self-attention code를 구현해보자!

    def calculate_attention(self, query, key, value, mask):
        # query, key , value : (n_batch, seq_len, d_k)
        # mask : (n_batch, seq_len, seq_len)
        d_k = key.shqpe[-1]
        attention_score = torch.matmul(query, key.transpose(-2,-1))
        # Q * K^T (n_batch, seq_len, seq_len)
        attention_score = attention_score/math.sqrt(d_k)
        if mask is not None:
            attention_score= attention_score.masked_fill(mask ==0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1)
        # (n_batch, seq_len, seq_len)
        out = torch.matmul(attention_prob, value)
        # (n_batch, seq_len, d_k)
        return out
  
        
    # 가장 중요한 부분이 forward()이므로 꼭 이해하고 넘어가자!!
    
    # def forward(self, *args, query, key, value, mask=None):
    def forward(self, query, key, value, mask=None): 
        # 굳이 q,k,v를 각각의 인자로 받는 이유는 self-attention 뿐만아니라, Cross-attention에도 활용하기 위함이다
        n_batch = query.size(0)
        
        def transform(x,fc):
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.model//self.h)
            out = out.transpose(1,2)
            return out
                    
        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)
        
        out = self.calculate_attention(query, key, value, mask)
        out = out.transpose(1,2)
        out = out.contiguous().view(n_batch, -1, self.d_model)
        out = self.out_fc(out)
        return out
