import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from layers.attention import Attention, squeeze_embedding, NoQueryAttention
from layers.laynorm import LayerNorm
from layers.squeeze import SqueezeEmbedding, DynamicLSTM


class EncoderPosition(nn.Module):
    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 hidden_size,
                 ):
        super().__init__()

        # atrribute
        self.hidden_size = hidden_size
        # gru
        gru_input_size = word_embed_dim + position_embed_dim  # word+pos
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(gru_input_size, self.hidden_size, batch_first=True, bidirectional=True)

    def forward(self, word, position, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = torch.cat((word, position), dim=-1)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        return gru_out


class EncoderPositionPOS(nn.Module):
    def __init__(self,
                 word_embedding,
                 position_size,
                 pos_size,
                 hidden_size,
                 ):
        super().__init__()

        # atrribute
        self.word_embedding = word_embedding  # word_embedding
        self.word_embedding_size = word_embedding.embedding_dim
        self.position_embedding_size = position_size
        self.pos_embedding_size = pos_size
        self.hidden_size = hidden_size

        # gru
        gru_input_size = self.word_embedding_size + \
                         + self.pos_embedding_size + \
                         self.position_embedding_size  # word+pos
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(gru_input_size, self.hidden_size, batch_first=True, bidirectional=True)

    def forward(self, context, position, pos, len_x):
        # word/pos/polar: batch,MAX_LEN  ;len_x:batch
        word_embedding = self.word_embedding(context)  # batch,MAX_LEN,word_embed_size
        x = torch.cat((word_embedding, position, pos), dim=-1)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        return gru_out
