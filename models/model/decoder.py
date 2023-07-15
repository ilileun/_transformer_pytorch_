import copy
import torch.nn as nn   


# decoder에서는 subseqent + pad_mask도 구해야 하므로, 
# forward()호출할때 역시 변경되어야 함

class Decoder(nn.Module):
    
    def __init__(self, decoder_block, n_layer, norm):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.norm = norm
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
    
    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        out = self.norm(out) #jieun_add
        return out
    