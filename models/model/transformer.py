import copy 
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Transformer(nn.Module):
    # generator은 decoder output의 마지막 dimension을 len(vocab)으로 변경
    # 실제 vocabulary 내 token에 대응시킬 수 있도록 shape을 만들어줘야 한다. 
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        
    
    def encoder(self,src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decoder(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask,src_tgt_mask)
    
    # src_embed(src), tgt_embed(tgt)와 같이 input을 transformerembedding으로 감싸준다!
    
    def forward(self, src, tgt):
        # 역시 mask인자를 넘겨줘야 한다!!
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, tgt_mask,src_tgt_mask)
        
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        # softmax를 이용해 vocabulary에 대한 확률값으로 변환 시켜 줌 
        # 마지막 dim = -1은 마지막 dimension인 len(vocab)에 대한 확률값으로 구해야하기 때문임
        return out, decoder_out
    
    # pad masking 생성해보자!

    def make_pad_mask(self, query, key, pad_idx=1):
        query_seq_len, key_seq_len = query.size(1), key.size(1)
        
        # (n_batch, 1,1, key_seq_len)
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        # (n_batch, 1, query_seq_len, key_seq_len)
        key_mask = key_mask.repeat(1,1,query_seq_len,1)
        
        # (n_batch, 1, query_seq_len, 1)
        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)
        # (n_batch, 1, query_seq_len, key_seq_len)
        query_mask = query_mask.repeat(1,1,1,key_seq_len)
        
        mask = key_mask & query_mask
        mask.require_grad = False
        
        return mask

    # 지금까지 encoder에서 다뤘던 Pad maskingdms 모두 동일한 문장 내에서 이루어 지는
    # self-attention이었음! (query, key가 동일)
    # 하지만 corss-attention의 경우 아님 (query(source), key(target)가 다름!)
    
    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask
            
    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask

    #decoder input에는 context와 sentence가 있음
    #context는 encoder에서 생성된것... 

    #decoder의 input에 추가적으로 들어오는 sentec를 이해하기 위해서는
    # Teacher Forcing이라는 개념이 중요하다!
    # 초기 학습시에는 엉터리 값을 내보내므로 teacher forcing을 이용해 label data를 input으로 활용!
    # 즉, 학습 과정에서 model이 생성해낸 token을 다음 token 생성 때 사용하는게 아니라,,
    # 실제 label data(ground truth)의 token을 사용하자! 

    # 이렇게 하면 모델의 학습성능을 비약적으로 향상시킬 수 있다

    # 주의 할점은 병렬 연산을 위해 ground truth embedding을 matrix로 만들어 input으로 사용하면
    # decoder에서 self-attention연산을 수행하게 될때 출력해야하는 token의 정답까지 알 수 있는 상황이
    # 발생하며 masking을 꼭 적용해줘야 한다. 
    # 이러한 masking기법은 subsequent masking이라고 한다.

    def make_subsequent_mask(self, query, key):
        query_seq_len, key_seq_len = query.size(1), key.size(1)
        
        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=-1).astype('unint8')
        mask = torch.tensor(tril, dtype = torch.bool, requires_grad = False, device = query.device)
        return mask

    def make_tgt_mask(self,tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask
        return pad_mask & seq_mask
