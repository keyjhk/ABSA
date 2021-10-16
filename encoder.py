import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from layers.attention import Attention, squeeze_embedding, NoQueryAttention
from layers.laynorm import LayerNorm
from layers.squeeze import SqueezeEmbedding, DynamicLSTM


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

        # dropout
        self.x_drop = nn.Dropout(p=0.3)

        # gru
        gru_input_size = self.word_embedding_size + self.pos_embedding_size  # word+pos
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(gru_input_size, self.hidden_size, batch_first=True, bidirectional=True,
                          num_layers=2)

    def forward(self, context, pos, len_x):
        # word/pos/polar: batch,MAX_LEN  ;len_x:batch
        word_embedding = self.word_embedding(context)  # batch,MAX_LEN,word_embed_size
        pos_embedding = self.pos_embedding(pos)  # batch,MAX_LEN,pos_embed_size

        x = torch.cat((word_embedding, pos_embedding),
                      dim=-1)  # batch,MAX_LEN,embed_size(word+pos+polar)
        # x = self.x_drop(x)
        # x = word_embedding # cancal pos_embedding
        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        return gru_out


class ATAEEncoder(nn.Module):
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
        gru_input_size = self.word_embedding_size + self.word_embedding_size  # word+aspect
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(gru_input_size, self.hidden_size, batch_first=True, bidirectional=True,
                          num_layers=2)

    def forward(self, context, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN  ;len_x:batch
        # aspct :  batch,1,embed_dim
        word_embedding = self.word_embedding(context)  # batch,MAX_LEN,word_embed_size
        # pos_embedding = self.pos_embedding(pos)  # batch,MAX_LEN,pos_embed_size

        x = torch.cat((word_embedding, aspect.expand(-1, word_embedding.shape[1], -1)),
                      dim=-1)  # batch,MAX_LEN,embed_size(word+pos+polar)
        # x = self.x_drop(x)
        # x = word_embedding # cancal pos_embedding
        # word+aspect to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        return gru_out


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

        # text_indices, aspect_indices = inputs[0], inputs[1]
        # x_len = torch.sum(text_indices != 0, dim=-1)  # batch
        # x_len_max = torch.max(x_len)  # 计算一个batch 中实际最长序列
        # aspect_len = torch.sum(aspect_indices != 0, dim=-1).float()  # batch
        #
        # x = self.embed(text_indices)  # batch,MAX_LEN,embed_dim
        #
        # # MAX_LEN ==> max_len ,从 人为规定的长度 切割为 实际最长的长度 底层调用了 packed_pad
        # # squeeze_embed 压缩 MAX_LEN 维度
        # x = self.squeeze_embedding(x, x_len)  # batch,MAX_LEN,embed_dim ==> batch,max_len,embed_dim
        # aspect = self.embed(aspect_indices)  # batch,MAX_LEN ==> batch,MAX_LEN,embed_dim
        # # 这一步是取aspect的平均向量 来源于论文的做法  sum/len
        # # batch,embed_dim [DIV] batch,1 ==> batch,embed_dim
        # aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        # # batch,1,embed_dim ==> batch,max_len,embed_dim  扩展时间步，便于和输入拼接
        # aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)  # aspect embedding
        # x = torch.cat((aspect, x), dim=-1)  # batch,max_len,embed_dim*2
        #
        # # batch,max_len,hidden_size
        # h, (_, _) = self.lstm(x, x_len)
        # # hidden 隐藏向量 和 aspect向量再作一次拼接
        # ha = torch.cat((h, aspect), dim=-1)  # batch,max_len,hidden_size*2
        #
        # # ha
        # _, score = self.attention(ha)  # score:batch,1,max_len  ; batch,max_len,hidden_size
        # output = torch.squeeze(torch.bmm(score, h), dim=1)  # batch,hidden_size
        #
        # out = self.dense(output)  # batch,hidden_size ==> batch,polarity_dim
        # return out


class ATAE_LSTMO(nn.Module):
    def __init__(self, num_polar, word_embedding, hidden_size):
        super().__init__()
        self.embed = word_embedding
        hidden_dim = hidden_size
        embed_dim = word_embedding.embedding_dim
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(embed_dim * 2, hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(hidden_dim + embed_dim, score_function='bi_linear')  # origin
        # self.attention = NoQueryAttention(opt.hidden_dim + opt.embed_dim, score_function='dot_product') # origin
        self.dense = nn.Linear(hidden_dim, num_polar)

    def forward(self, context, aspect_pool):
        # context: batch,MAX_LEN  ; aspect_indices:batch,MAX_LEN 用于表示每个batch出现的aspect
        # MAX_LEN 是 sequence最长的长度
        x_len = torch.sum(context != 0, dim=-1)  # batch
        x_len_max = torch.max(x_len)  # 计算一个batch 中实际最长序列

        x = self.embed(context)  # batch,MAX_LEN,embed_dim
        # MAX_LEN ==> max_len ,从 人为规定的长度 切割为 实际最长的长度 底层调用了 packed_pad
        # squeeze_embed 压缩 MAX_LEN 维度
        x = self.squeeze_embedding(x, x_len)  # batch,MAX_LEN,embed_dim ==> batch,max_len,embed_dim

        # batch,1,embed_dim ==> batch,max_len,embed_dim  扩展时间步，便于和输入拼接
        aspect = aspect_pool.expand(-1, x_len_max, -1)  # aspect embedding
        x = torch.cat((aspect, x), dim=-1)  # batch,max_len,embed_dim*2

        # batch,max_len,hidden_size
        h, (_, _) = self.lstm(x, x_len)
        # hidden 隐藏向量 和 aspect向量再作一次拼接
        ha = torch.cat((h, aspect), dim=-1)  # batch,max_len,hidden_size*2

        # ha
        _, score = self.attention(ha)  # score:batch,1,max_len  ; batch,max_len,hidden_size
        output = torch.squeeze(torch.bmm(score, h), dim=1)  # batch,hidden_size

        out = self.dense(output)  # batch,hidden_size ==> batch,polarity_dim
        return out
