import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from layers.attention import squeeze_embedding, NoQueryAttention


class ATAE_LSTM(nn.Module):
    def __init__(self, num_polar, word_embedding, hidden_size):
        super(ATAE_LSTM, self).__init__()
        self.word_embedding = word_embedding
        word_embed_dim = word_embedding.embedding_dim
        self.lstm = nn.LSTM(word_embed_dim * 2, hidden_size, batch_first=True)

        self.attention = NoQueryAttention(hidden_size + word_embed_dim, score_function='bi_linear')  # origin
        self.dense = nn.Linear(hidden_size, num_polar)

    def forward(self, context, aspect, len_x):
        # text_indices: batch,MAX_LEN  ; aspect_indices:batch,MAX_LEN 用于表示每个batch出现的aspect
        # MAX_LEN 是 sequence最长的长度

        # context, aspect, len_x
        max_x = len_x.max()
        x = self.word_embedding(context)  # batch,max_len,word_dim
        x = squeeze_embedding(x, len_x)  # batch,seq_len,word_dim
        x = torch.cat((x, aspect.expand(-1, max_x, -1)), dim=-1)  # batch,seq_len,word_dim*2

        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        h, (_, _) = self.lstm(pad_x)
        h, _ = pad_packed_sequence(h, batch_first=True)  # batch,seq_len,hidden_dim

        ha = torch.cat((h, aspect.expand(-1, max_x, -1)), dim=-1)  # batch,max_len,hidden_size
        _, score = self.attention(ha)  # score:batch,1,max_len  ; batch,max_len,hidden_size
        output = torch.squeeze(torch.bmm(score, h), dim=1)  # batch,hidden_size

        out = self.dense(output)  # batch,hidden_size ==> batch,polarity_dim
        return out
