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


class EncoderPositionPolar(nn.Module):
    def __init__(self,
                 word_size,
                 position_size,
                 polar_size,
                 hidden_size,
                 ):
        super().__init__()
        # gru
        gru_input_size = word_size + polar_size + position_size
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(gru_input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, context, position, polar, len_x):
        # word/pos/polar: batch,MAX_LEN  ;len_x:batch
        x = torch.cat((context, position, polar), dim=-1)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        return gru_out


class EncoderPolarPosition(nn.Module):
    def __init__(self,
                 polar_size,
                 position_size,
                 hidden_size,
                 ):
        super().__init__()
        # gru
        gru_input_size = polar_size + position_size
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(gru_input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, polar, position, len_x):
        # word/pos/polar: batch,MAX_LEN  ;len_x:batch
        x = torch.cat((position, polar), dim=-1)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        return gru_out


class Test(nn.Module):
    '''
    polar+posiiton ：自注意力 作用在h上
    word+position：gru
    拼接
    res:
        loss:0.5274 acc:78.23% f1:64.380936
        no-sp：loss:0.5318 acc:80.20% f1:72.143256
    lap:
        no-sp:loss:0.6269 acc:74.73% f1:70.424389

    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        # atrribute
        self.hidden_size = hidden_dim
        # gru
        gru_input_size = word_embed_dim + position_embed_dim  # word+pos
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(gru_input_size, self.hidden_size, batch_first=True, bidirectional=True)

        # decoder
        hidden_dim = hidden_dim * 2  # bidirection
        self.aspect_u = nn.Linear(word_embed_dim, word_embed_dim)
        # 情感词自注意力
        self.attention_p = Attention(polar_embed_dim + position_embed_dim)
        # 情感词注意力权重
        self.attention_hp = NoQueryAttention(hidden_dim + position_embed_dim + polar_embed_dim,
                                             score_function='bi_linear')
        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim + position_embed_dim,
                                              score_function='bi_linear')  # aspect + context

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_polar)
        )

    def forward(self, word, position, polar, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = torch.cat((word, position), dim=-1)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        encoder_out = gru_out
        # decoder
        # squeeze
        max_x = len_x.max()
        position = squeeze_embedding(position, len_x.cpu())
        polar = squeeze_embedding(polar, len_x.cpu())
        # sp
        polar = torch.cat((polar, position), dim=-1)
        polar = torch.tanh(self.attention_p(k=polar, q=polar)[0])
        hp = torch.cat((encoder_out, polar), dim=-1)
        _, scores = self.attention_hp(hp)
        sp = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # sa
        hap = torch.cat((encoder_out, position, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size

        # concat
        s = torch.cat((s_max, sp), dim=-1)

        out = self.dense(s)
        return out  # batch,num_polar


class Test1(nn.Module):
    '''
    word+position+polar：gru
    拼接
    res:
        loss:0.5000 acc:80.23% f1:69.581295
        loss:0.5297 acc:80.14% f1:68.744183
    lap:
        loss:0.7497 acc:74.64% f1:70.492448
        loss:0.6720 acc:74.86% f1:69.943652
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        # atrribute
        self.hidden_size = hidden_dim
        # gru
        gru_input_size = word_embed_dim + position_embed_dim + polar_embed_dim  # word+pos
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(gru_input_size, self.hidden_size, batch_first=True, bidirectional=True)

        # decoder
        hidden_dim = hidden_dim * 2  # bidirection
        self.aspect_u = nn.Linear(word_embed_dim, word_embed_dim)
        # 情感词自注意力
        self.attention_p = Attention(polar_embed_dim + position_embed_dim)
        # 情感词注意力权重
        self.attention_hp = NoQueryAttention(hidden_dim + position_embed_dim + polar_embed_dim,
                                             score_function='bi_linear')
        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim + position_embed_dim,
                                              score_function='bi_linear')  # aspect + context

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_polar)
        )

    def forward(self, word, position, polar, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = torch.cat((word, position, polar), dim=-1)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        encoder_out = gru_out
        # decoder
        # squeeze
        max_x = len_x.max()
        position = squeeze_embedding(position, len_x.cpu())
        polar = squeeze_embedding(polar, len_x.cpu())
        # sp
        # polar = torch.cat((polar,position),dim=-1)
        # polar, _ = self.attention_p(k=polar, q=polar)
        # polar = torch.tanh(polar)
        # hp = torch.cat((encoder_out, polar), dim=-1)
        # _, scores = self.attention_hp(hp)
        # sp = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # sa
        hap = torch.cat((encoder_out, position, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size

        # concat
        s = torch.cat((s_max, sa), dim=-1)

        out = self.dense(s)
        return out  # batch,num_polar


class Test2(nn.Module):
    '''
    模仿imn
    polar+posiiton ：gru ,拼接上下文的平均向量,作为注意力计算然后加权到polar
    word_position：gru
    拼接 polar + word
    res:
        loss:0.5184 acc:78.30% f1:63.956905
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        # atrributes
        self.hidden_dim = hidden_dim
        self.polar_dim = polar_embed_dim
        # gru
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(word_embed_dim + position_embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_p = nn.GRU(polar_embed_dim + position_embed_dim, polar_embed_dim, batch_first=True, bidirectional=True)
        # decoder

        hidden_dim = hidden_dim * 2  # bidirection
        polar_embed_dim = polar_embed_dim * 2
        self.aspect_u = nn.Linear(word_embed_dim, word_embed_dim)

        # 情感词注意力权重 h+p
        self.attention_hp = NoQueryAttention(hidden_dim + polar_embed_dim,
                                             score_function='bi_linear')
        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim + position_embed_dim,
                                              score_function='bi_linear')  # aspect + context

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim + polar_embed_dim, hidden_dim),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_polar)
        )

    def forward(self, word, position, polar, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = torch.cat((word, position), dim=-1)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2
        encoder_word = gru_out  # batch,seq_len,word_dim

        # decoder
        polar = torch.cat((polar, position), dim=-1)
        pad_p = pack_padded_sequence(polar, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru_p(pad_p)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2
        encoder_polar = gru_out  # batch,seq_len,polar_dim

        # squeeze
        max_x = len_x.max()
        position = squeeze_embedding(position, len_x)
        mw = torch.mean(encoder_word, dim=1).unsqueeze(1)  # batch,1,word_dim
        mp = torch.mean(encoder_polar, dim=1).unsqueeze(1)  # batch,1,polar_dim
        # sp
        hp = torch.cat((encoder_polar, mw.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hp(hp)
        sp = torch.bmm(scores, encoder_polar).squeeze(1)  # batch,polar_dim
        # sa
        hap = torch.cat((encoder_word, position, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_word).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_word, dim=1)[0].squeeze(1)  # batch,hidden_size
        # concat
        s = torch.cat((sa, sp), dim=-1)

        out = self.dense(s)
        return out  # batch,num_polar


class Test3(nn.Module):
    '''
    word+position+polar：gru
    是test1的改版  我们在position-aware的时候 增加一个情感词关注 h,polar,aspect
    拼接
    res:
       test1:  loss:0.5000 acc:80.23% f1:69.581295
       loss:0.5066 acc:80.77% f1:71.684482
    laptop:
        test1：loss:0.7497 acc:74.64% f1:70.492448
        loss:  0.6262 acc:74.39% f1:69.309383
        smax ==> hn

    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        # atrribute
        self.hidden_size = hidden_dim
        # gru
        gru_input_size = word_embed_dim + position_embed_dim + polar_embed_dim  # word+pos
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(gru_input_size, self.hidden_size, batch_first=True, bidirectional=True)

        # decoder
        hidden_dim = hidden_dim * 2  # bidirection
        self.aspect_u = nn.Linear(word_embed_dim, word_embed_dim)
        # 情感词自注意力
        self.attention_p = Attention(polar_embed_dim + position_embed_dim)
        # 情感词注意力权重
        self.attention_hsp = NoQueryAttention(word_embed_dim + hidden_dim + polar_embed_dim,
                                              score_function='bi_linear')
        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim + position_embed_dim,
                                              score_function='bi_linear')  # aspect + context

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_polar)
        )

    def forward(self, word, position, polar, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = torch.cat((word, position, polar), dim=-1)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        encoder_out = gru_out
        # decoder
        # squeeze
        max_x = len_x.max()
        position = squeeze_embedding(position, len_x.cpu())
        polar = squeeze_embedding(polar, len_x.cpu())
        # sp
        hsp = torch.cat((encoder_out, polar, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hsp(hsp)
        sp = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # sa
        hap = torch.cat((encoder_out, position, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size
        hn = encoder_out[:, -1, :]
        # concat
        s = torch.cat((s_max, sa, sp), dim=-1)

        out = self.dense(s)
        return out  # batch,num_polar


class Test4(nn.Module):
    '''
    模仿imn
    polar+posiiton ：self-attention ,拼接上下文的平均向量,作为注意力计算然后加权到polar
    word_position：gru
    拼接 polar + word
    res:

    lap:
        loss:0.6645 acc:71.72% f1:66.510145 no-hn
        loss:0.6647 acc:73.23% f1:68.341442 hn

    使用lcf-bert的方式提取情感词的特征 拼接 权重作用在h上 最后拼接
        res:loss:0.5128 acc:78.45% f1:66.190949
        lap:loss:0.6879 acc:74.23% f1:69.587206
    使用lcf-bert的方式提取情感词的特征 拼接 权重作用在polar上 最后拼接



    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        # atrributes
        self.hidden_dim = hidden_dim
        self.polar_dim = polar_embed_dim
        # gru
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(word_embed_dim + position_embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # decoder

        hidden_dim = hidden_dim * 2  # bidirection

        # 情感词注意力权重 h+p
        self.attention_p = Attention(polar_embed_dim + position_embed_dim, n_head=2)
        self.pct_p = nn.Sequential(
            nn.Linear(polar_embed_dim + position_embed_dim,
                      polar_embed_dim + position_embed_dim),
            nn.ReLU(),
            nn.Linear(polar_embed_dim + position_embed_dim,
                      polar_embed_dim + position_embed_dim)
        )
        self.attention_p1 = Attention(polar_embed_dim + position_embed_dim, n_head=2)

        self.attention_hp = NoQueryAttention(hidden_dim + polar_embed_dim + position_embed_dim,
                                             score_function='bi_linear')
        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim + position_embed_dim,
                                              score_function='bi_linear')  # aspect + context

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_polar)
        )

    def forward(self, word, position, polar, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = torch.cat((word, position), dim=-1)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2
        encoder_out = gru_out  # batch,seq_len,word_dim

        # decoder

        # squeeze
        max_x = len_x.max()
        polar = squeeze_embedding(polar, len_x)
        position = squeeze_embedding(position, len_x)
        # self-attention for polar
        polar = torch.cat((polar, position), dim=-1)
        polar = torch.tanh(self.attention_p(polar, polar)[0])  # batch,seq_len,polar+position
        polar = self.pct_p(polar)  # batch,seq_len,polar+position
        # mean
        mp = torch.mean(polar, dim=1).unsqueeze(1)  # batch,1,polar+position
        mw = torch.mean(encoder_out, dim=1).unsqueeze(1)  # batch,1,hidden_dim
        # sp
        hp = torch.cat((encoder_out, mp.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hp(hp)
        sp = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_dim
        # sa
        hap = torch.cat((encoder_out, position, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size
        hn = encoder_out[:, -1, :]
        # concat
        s = torch.cat((sa, sp, s_max), dim=-1)

        out = self.dense(s)
        return out  # batch,num_polar


class Test4a(nn.Module):
    '''
    模仿imn
    polar+posiiton ：self-attention ,拼接上下文的平均向量,作为注意力计算然后加权到polar
    word_position：gru
    拼接 polar + word


    使用lcf-bert的方式提取情感词的特征 拼接 权重作用在polar上 最后拼接
        res：loss:0.5098 acc:80.45% f1:70.285988
             loss:0.5064 acc:79.46% f1:67.829853
        lap: loss:0.6254 acc:74.08% f1:68.389943
             loss:0.6362 acc:74.83% f1:71.605024

        再pct 基础上添加一层 mhsa  xxx
            res: loss:0.5144 acc:79.59% f1:67.726914
            lap:loss:0.6539 acc:73.89% f1:68.438174
        特征结合的方式下，是否需要smax
        no-smax:
            res:loss:0.5238 acc:78.68% f1:64.762679
            lap:loss:0.6540 acc:72.79% f1:68.741203
        no-sp:
            loss:0.5097 acc:80.00% f1:69.989330
            loss:0.6486 acc:73.42% f1:70.044324

    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        # atrributes
        self.hidden_dim = hidden_dim
        self.polar_dim = polar_embed_dim
        # gru
        # gru_input_size = self.word_embedding_size   # word
        self.gru = nn.GRU(word_embed_dim + position_embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # decoder

        hidden_dim = hidden_dim * 2  # bidirection

        # 情感词注意力权重 h+p
        self.attention_p = Attention(polar_embed_dim + position_embed_dim, n_head=2)
        self.pct_p = nn.Sequential(
            nn.Linear(polar_embed_dim + position_embed_dim,
                      polar_embed_dim + position_embed_dim),
            nn.ReLU(),
            nn.Linear(polar_embed_dim + position_embed_dim,
                      polar_embed_dim + position_embed_dim)
        )

        self.attention_hp = NoQueryAttention(hidden_dim + polar_embed_dim + position_embed_dim,
                                             score_function='bi_linear')
        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim + position_embed_dim,
                                              score_function='bi_linear')  # aspect + context

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2 + position_embed_dim + polar_embed_dim, hidden_dim),
            # + position_embed_dim+polar_embed_dim, hidden_dim
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_polar)
        )

    def forward(self, word, position, polar, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = torch.cat((word, position), dim=-1)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2
        encoder_out = gru_out  # batch,seq_len,word_dim

        # decoder

        # squeeze
        max_x = len_x.max()
        polar = squeeze_embedding(polar, len_x)
        position = squeeze_embedding(position, len_x)
        # self-attention for polar
        polar = torch.cat((polar, position), dim=-1)
        polar = torch.tanh(self.attention_p(polar, polar)[0])  # batch,seq_len,polar+position
        polar = self.pct_p(polar)  # batch,seq_len,polar+position
        # polar = torch.tanh(self.attention_p1(polar,polar)[0])
        # mean
        mp = torch.mean(polar, dim=1).unsqueeze(1)  # batch,1,polar+position
        mw = torch.mean(encoder_out, dim=1).unsqueeze(1)  # batch,1,hidden_dim
        # sp
        hp = torch.cat((polar, mw.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hp(hp)
        sp = torch.bmm(scores, polar).squeeze(1)  # batch,polar+posititon
        # sa
        hap = torch.cat((encoder_out, position, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size
        hn = encoder_out[:, -1, :]
        # concat
        s = torch.cat((sa, s_max, sp), dim=-1)

        out = self.dense(s)
        return out  # batch,num_polar


class Test5(nn.Module):
    '''
    模仿lcf-bert
    polar+posiiton ：self-attention ,拼接上下文的平均向量,作为注意力计算然后加权到polar
    word_position：self-attention
    拼接 polar + word
    res:

    lap:
      loss:1.0595 acc:53.45% f1:23.220974
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        # atrributes
        self.hidden_dim = hidden_dim
        self.polar_dim = polar_embed_dim
        # decoder

        # 情感词注意力权重 h+p
        n_head = 5
        # for context
        self.attention_c = Attention(word_embed_dim + position_embed_dim, n_head=5)
        self.pct_c = nn.Sequential(
            nn.Linear(word_embed_dim + position_embed_dim, word_embed_dim + position_embed_dim),
            nn.ReLU(),
            nn.Linear(word_embed_dim + position_embed_dim,
                      word_embed_dim + position_embed_dim)
        )
        self.attention_c1 = Attention(word_embed_dim + position_embed_dim, n_head=5)
        # for polar
        self.attention_p = Attention(polar_embed_dim + position_embed_dim, n_head=5)
        self.pct_p = nn.Sequential(
            nn.Linear(polar_embed_dim + position_embed_dim,
                      polar_embed_dim + position_embed_dim),
            nn.ReLU(),
            nn.Linear(polar_embed_dim + position_embed_dim,
                      polar_embed_dim + position_embed_dim)
        )
        self.attention_p1 = Attention(polar_embed_dim + position_embed_dim, n_head=5)

        # concat batch,seq,hidden+word+pos
        concat_dim = hidden_dim + polar_embed_dim + 2 * position_embed_dim
        self.fil = nn.Linear(concat_dim,
                             concat_dim)
        self.attention_fil = Attention(concat_dim,
                                       n_head=5)

        self.dense = nn.Sequential(
            nn.Linear(concat_dim,
                      hidden_dim),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_polar)
        )

    def forward(self, word, position, polar, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        # squeeze
        word = squeeze_embedding(word, len_x.cpu())
        position = squeeze_embedding(position, len_x.cpu())
        polar = squeeze_embedding(polar, len_x.cpu())

        # c
        c = torch.cat((word, position), dim=-1)  # w+p
        c = torch.tanh(self.attention_c(c, c)[0])  # w+p
        c = self.pct_c(c)  # w+p
        c = torch.tanh(self.attention_c1(c, c)[0])
        # p
        p = torch.cat((polar, position), dim=-1)
        p = torch.tanh(self.attention_p(p, p)[0])
        p = self.pct_p(p)
        p = torch.tanh(self.attention_p1(p, p)[0])

        # concat
        cp = torch.cat((c, p), dim=-1)
        cp = self.fil(cp)
        cp = torch.tanh(self.attention_fil(cp, cp)[0])
        cp = torch.max(cp, dim=1)[0]
        out = self.dense(cp)
        return out  # batch,num_polar
