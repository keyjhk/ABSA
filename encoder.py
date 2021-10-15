import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from layers.attention import Attention, squeeze_embedding
from layers.laynorm import LayerNorm


class Encoder(nn.Module):
    def __init__(self,
                 num_words,
                 num_pos,
                 num_polar,
                 word_embedding,
                 polar_embedding,
                 pos_embedding_size,
                 dropout_lab,
                 hidden_size,
                 device='cpu'
                 ):
        super().__init__()

        # atrribute
        self.num_words = num_words
        self.num_pos = num_pos
        self.num_polar = num_polar
        self.word_embedding = word_embedding  # word_embedding
        self.word_embedding_size = word_embedding.embedding_dim
        self.pos_embedding_size = pos_embedding_size
        self.pos_embedding = nn.Embedding(self.num_pos, self.pos_embedding_size)  # pos_embedding
        self.polar_embedding = polar_embedding  # polar_embedding
        self.polar_embedding_size = hidden_size  # polar_embedding_size 应该和hidden_size保持一致
        self.hidden_size = hidden_size
        self.device = device

        # gru
        gru_input_size = self.word_embedding_size + self.pos_embedding_size
        self.gru = nn.GRU(gru_input_size, self.hidden_size, batch_first=True, bidirectional=True,
                          num_layers=2, dropout=0.3)

        # attention layer
        self.att_senti = Attention(embed_dim=self.hidden_size * 2, score_function='scaled_dot_product')
        # fuse context with sentiment
        self.hs = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.1)
        )
        self.laynorm= LayerNorm(self.hidden_size)

    def self_attention(self, k):
        # q==k batch,seq_len,hidden_size
        scores = torch.bmm(k, k.permute(1, 2))  # batch,seq_len,seq_len
        scores = F.softmax(scores, dim=-1)
        return scores


    def forward(self, word, pos, polar, len_x, mask_x):
        # word/pos/polar: batch,max_len  ;len_x:batch
        word_embedding = self.word_embedding(word)  # batch,max_len,word_embed_size
        pos_embedding = self.pos_embedding(pos)  # batch,max_len,pos_embed_size
        polar_embedding = self.polar_embedding(polar)  # batch,max_len,polar_embed_size
        x = torch.cat((word_embedding, pos_embedding),
                      dim=-1)  # batch,max_len,embed_size(word+pos+polar)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        # attention for hidden/sentiment
        polar_embedding = squeeze_embedding(polar_embedding, len_x)  # batch,seq_len,hidden_size
        mask_x = squeeze_embedding(mask_x, len_x, padding_value=True)  # batch,seq_len
        _, score = self.att_senti(gru_out, gru_out, mask=mask_x)  # batch,seq_len,seq_len
        polar = torch.bmm(score, polar_embedding)  # bach,seq_len,hidden_size
        # fuse  batch,seq_len,hidden_size*3==>batch,seq_len,hidden_size
        hs = self.hs(torch.cat((gru_out, polar), dim=-1))
        # hs = self.laynorm(hs)
        return hs, hidden
