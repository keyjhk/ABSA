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
                 pos_embedding,
                 hidden_size,
                 ):
        super().__init__()

        # atrribute
        self.num_words = num_words
        self.num_pos = num_pos
        self.num_polar = num_polar
        self.word_embedding = word_embedding  # word_embedding
        self.word_embedding_size = word_embedding.embedding_dim
        self.pos_embedding = pos_embedding
        self.pos_embedding_size = pos_embedding.embedding_dim
        self.hidden_size = hidden_size

        # gru
        gru_input_size = self.word_embedding_size + self.pos_embedding_size  # word+pos
        self.gru = nn.GRU(gru_input_size, self.hidden_size, batch_first=True, bidirectional=True,
                          num_layers=2)

    def forward(self, context, pos, len_x):
        # word/pos/polar: batch,MAX_LEN  ;len_x:batch
        word_embedding = self.word_embedding(context)  # batch,MAX_LEN,word_embed_size
        pos_embedding = self.pos_embedding(pos)  # batch,MAX_LEN,pos_embed_size

        x = torch.cat((word_embedding, pos_embedding),
                      dim=-1)  # batch,MAX_LEN,embed_size(word+pos+polar)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        return gru_out
