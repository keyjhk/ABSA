import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from layers.attention import Attention, squeeze_embedding, NoQueryAttention
from data_utils import SOS_TOKEN


class AttentionWindow(nn.Module):
    def __init__(self, hidden_size, wsize=4, eye=True):
        super().__init__()
        self.attention_w = Attention(hidden_size)
        self.wsize = wsize
        self.eye = eye

    def window_mask(self, repr):
        wsize = self.wsize
        batch, seq_len, _ = repr.shape
        device = repr.device
        _mask = torch.ones(seq_len, seq_len)
        u_mask = torch.triu(_mask, wsize + 1).type(torch.bool)  # under triange is zero
        l_mask = torch.tril(_mask, -(wsize + 1)).type(torch.bool)
        if self.eye:  # vision itself
            mask = u_mask ^ l_mask
        else:
            mask = u_mask ^ l_mask ^ torch.eye(seq_len, dtype=torch.bool)

        mask = mask.unsqueeze(0).expand(batch, -1, -1)  # batch,seq_len,seq_len

        return mask.to(device)  # batch,seq_len,seq_len

    def forward(self, repr):
        # repr:batch,seq,hidden_size ss
        mask = self.window_mask(repr)
        _, scores = self.attention_w(k=repr, q=repr, mask=mask)
        ra = torch.bmm(scores, repr)  # batch,seq,hidden_size

        return repr + torch.relu(ra)


class LocationDecoder(nn.Module):
    def __init__(self,
                 word_embedding,
                 hidden_dim,
                 tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_dim
        self.word_embedding = word_embedding
        self.l = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention_ch = Attention(hidden_dim)  # context /hidden
        self.decoder = nn.GRU(word_embedding.embedding_dim,
                              hidden_dim, batch_first=True)

        self.dense = nn.Linear(hidden_dim * 2, self.tokenizer.max_seq_len)

    def forward(self, encoder_out, last_hidden, context_indices):
        # encoder: batch,seq_len,hidden_size*2 ;
        # context_indices : batch,MAX_LEN
        # last_hidden: 2(directions),batch,hidden_size
        mb = encoder_out.shape[0]
        device = encoder_out.device
        sos_idx = self.tokenizer.word2idx[SOS_TOKEN]
        sos = torch.tensor([sos_idx] * mb, device=device).view(-1, 1)  # batch,1
        sos = self.word_embedding(sos)  # batch,1,word_dim
        decoder_out = sos
        # last_hidden from encoder, merge forward/backward
        last_hidden = last_hidden[0] + last_hidden[1]  # batch,hidden
        last_hidden = last_hidden.unsqueeze(0)  # 1, batch,hidden_size
        # encoder ,merge forward/backward
        # batch,seq_len,hidden_size
        encoder_out = encoder_out[:, :, :self.hidden_size] + encoder_out[:, :, self.hidden_size:]
        encoder_out = self.l(encoder_out)
        locations = []

        for i in range(2):  # sos,as,ae
            # decoder_out: batch,1,hidden_size ; last_hidden: 1,batch,hidden_size
            decoder_out, last_hidden = self.decoder(decoder_out, last_hidden)

            weighted_context, _ = self.attention_ch(k=encoder_out, q=decoder_out)  # batch,1,hidden_size
            wch = torch.cat((decoder_out, weighted_context), dim=-1)
            location = self.dense(wch)  # batch,1,max_len
            location = location.squeeze(1)  # batch,max_len
            locations.append(location)  # probability
            # max prob location
            predict_location = torch.argmax(location.softmax(dim=-1), dim=-1, keepdim=True)  # batch,1 ;1 means index
            decoder_out = torch.gather(context_indices, 1, predict_location)  # batch,1 ;posiiton in context
            decoder_out = self.word_embedding(decoder_out)  # batch,1,word_dim
        return torch.stack(locations, dim=1)  # batch,2,seq_len


class PolarDecoder(nn.Module):
    def __init__(self,
                 word_embedding,
                 hidden_size,
                 num_polar,
                 tokenizer,
                 name):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.word_embedding = word_embedding
        self.num_polar = num_polar


        self.attention_ch = Attention(hidden_size)  # hidden/context score_function='mlp'
        word_dim = word_embedding.embedding_dim
        self.decoder = nn.GRU(word_dim,
                              hidden_size, batch_first=True)

        self.dense = nn.Linear(hidden_size * 3, num_polar)

    def forward(self, encoder_out, last_hidden):
        # encoder: batch,seq_len,hidden_size*2 ;
        # aspect : # batch,1,embed_dim
        # last_hidden: 2(directions),batch,hidden_size
        mb ,seq = encoder_out.shape[0],encoder_out.shape[1]

        device = encoder_out.device

        # init decoder input
        sos_idx = self.tokenizer.word2idx[SOS_TOKEN]
        sos = torch.tensor([sos_idx] * mb, device=device).view(-1, 1)  # batch,1
        sos = self.word_embedding(sos)  # batch,1,word_dim
        decoder_out = sos
        # int decoder hidden
        # last_hidden from encoder, merge forward/backward
        last_hidden = last_hidden[0] + last_hidden[1]  # batch,hidden
        last_hidden = last_hidden.unsqueeze(0)  # 1, batch,hidden_size

        # decoder
        # decoder_out: batch,1,hidden_size  ; encoder: batch,seq,hidden_size
        encoder_out = encoder_out[:, :, :self.hidden_size] + encoder_out[:, :, self.hidden_size:]
        decoder_out, _ = self.decoder(decoder_out, last_hidden)
        _, score = self.attention_ch(q=decoder_out, k=encoder_out)  # batch,1,seq_len
        context = torch.bmm(score,encoder_out)  # batch,1,hidden_size

        smax = torch.max(encoder_out,dim=1,keepdim=True)[0]  # batch,1,hidden_size
        # ch = torch.cat((context, decoder_out), dim=-1)  # batch,1,hidden_size*2
        ch = torch.cat((context, decoder_out,smax), dim=-1)  # batch,1,hidden_size*3
        ch = ch.squeeze(1)  # batch,hidden_size*2


        out = self.dense(ch)  # batch,num_polar

        return out


class EncoderPosition(nn.Module):
    '''
    论文的版本 只考虑位置信息得到编码
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 hidden_size,
                 ):
        super().__init__()

        # atrribute
        self.name = 'encoder-position'
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

        return gru_out, hidden


class BiEncoder(nn.Module):
    '''
    类似atae 考虑位置编码 直接作aspect的注意力计算
    只不过注意力计算的时候不考虑position

    word：gru
    no position aware;aspect+hidden做注意力计算
    sa+hn：
        res:
            loss:0.5470 acc:77.54% f1:65.686826
        lap:
            loss:0.6908 acc:71.96% f1:66.170036

    + position-aware;sa + hn;aspect+position+hidden做注意力计算
    sa+hn
        res：
            loss:0.5213 acc:79.15% f1:68.063962
        lap:
            loss:0.6579 acc:71.88% f1:66.236789

    + no-position-aware
    smax + hn 丢弃了所谓的位置感知注意力
        res:
            loss:0.4988 acc:80.71% f1:70.184447
        lap:
            loss:0.6370 acc:74.36% f1:70.390545
        对比 不考虑位置编码的模型
        smax + hn: 融合位置编码的aspect 中显示 smax最有用
                最大池化发生了作用
        res：
            loss:0.6088 acc:79.17% f1:70.223417
        lap:
            loss:0.6551 acc:73.01% f1:68.514047
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        self.name = 'testb'
        # atrribute
        self.hidden_size = hidden_dim
        # gru
        gru_input_size = word_embed_dim + position_embed_dim  # word
        self.gru = nn.GRU(gru_input_size, self.hidden_size,
                          batch_first=True, bidirectional=True)
        # decoder
        hidden_dim = hidden_dim * 2  # bidirection

        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim + position_embed_dim,
                                              score_function='bi_linear')  # aspect + context

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_polar)
        )

    def forward(self, word, position, polar, pos, aspect, len_x):
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

        # sa
        hap = torch.cat((encoder_out, position, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size
        hn = encoder_out[:, -1, :]  # batch,hidden_size
        # concat
        # s = torch.cat((s_max, sp, sa), dim=-1)
        # s = torch.cat((s_max, sa, hn), dim=-1)
        # s = torch.cat((sa, hn), dim=-1)  # atae
        s = torch.cat((s_max, hn), dim=-1)  # atae

        out = self.dense(s)
        return out, gru_out, hidden  # batch,num_polar


class BilayerEncoder(nn.Module):
    '''
    第一层 word
    第二层 +position
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 hidden_size
                 ):
        super().__init__()
        self.name = 'bilayer-encoder'
        # atrribute
        self.word_embed_dim = word_embed_dim
        self.hidden_size = hidden_size
        # gru
        # gru_input_size = self.word_embedding_size   # word
        self.uni_gru = nn.GRU(word_embed_dim, hidden_size,
                              batch_first=True, bidirectional=True)
        self.bi_gru = nn.GRU(hidden_size * 2 + position_embed_dim,
                             hidden_size,
                             batch_first=True, bidirectional=True)

    def forward(self, word, position, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = word
        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        uni_out, uni_hidden = self.uni_gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        uni_out, _ = pad_packed_sequence(uni_out, batch_first=True)  # batch,seq_len,hidden_size*2
        # uni + position
        # bi_in = uni_out[:, :, :self.hidden_size] + uni_out[:, :, self.hidden_size:]
        bi_in = uni_out
        bi_in = torch.cat((bi_in, position), dim=-1)  # + position
        bi_out, bi_hidden = self.bi_gru(bi_in)

        return bi_out, uni_out


class BilayerEncoderP(nn.Module):

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 hidden_size
                 ):
        super().__init__()
        self.name = 'bilayer-encoderp'
        # atrribute
        self.word_embed_dim = word_embed_dim
        self.hidden_size = hidden_size
        # gru
        # gru_input_size = self.word_embedding_size   # word
        self.uni_gru = nn.GRU(word_embed_dim + position_embed_dim, hidden_size,
                              batch_first=True, bidirectional=True)
        self.bi_gru = nn.GRU(hidden_size * 2,
                             hidden_size,
                             batch_first=True, bidirectional=True)
        self.uni_dropout = nn.Dropout(p=0.5)
        self.bi_dropout = nn.Dropout(p=0.5)
        # self attention
        self.atention_window = AttentionWindow(hidden_size * 2)

    def forward(self, word, position, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = torch.cat((word, position), dim=-1)
        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        uni_out, uni_hidden = self.uni_gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        uni_out, _ = pad_packed_sequence(uni_out, batch_first=True)  # batch,seq_len,hidden_size*2
        # uni_out = self.uni_dropout(uni_out)
        # uni + position
        # bi_in = uni_out[:, :, :self.hidden_size] + uni_out[:, :, self.hidden_size:]
        bi_in = uni_out
        bi_out, bi_hidden = self.bi_gru(bi_in)  # batch,seq_len,hidden_size * 2
        bi_out = self.bi_dropout(bi_out)
        # bi_out, _ = self.bi_sa(k=bi_out, q=bi_out)  # batch,seq_len,hidden_size * 2
        ua = self.atention_window(uni_out)
        # uni_out = ua
        # return bi_out, uni_out
        return uni_out, uni_hidden  # for decoder


class BilayerPrimary(nn.Module):

    def __init__(self,
                 word_embed_dim,
                 hidden_dim,
                 num_polar):
        super().__init__()
        # decoder
        self.name = 'bilayer-primary'

        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim,
                                              score_function='bi_linear')

        self.output_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_layer1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_w = AttentionWindow(hidden_dim)
        self.dense = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Tanh(),  # tanh,relu
            nn.Linear(hidden_dim, num_polar),
        )

    def project(self, repr, aspect, len_x, aspect_boundary=None):
        # repr: batch,seq_len,hidden_size
        max_x = len_x.max()
        # repr = self.e(repr)
        # sa
        hap = torch.cat((repr, aspect.expand(-1, max_x, -1)),
                        dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, repr).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(repr, dim=1)[0].squeeze(1)  # batch,hidden_size
        # sw
        if aspect_boundary != None:
            sw = self.attention_w(repr)  # batch,seq_len,hidden_size
            aspect_boundary = aspect_boundary[:, 1]  # batch,the end aspect
            _sw = []
            for i in range(aspect_boundary.shape[0]):
                _sw.append(sw[i, aspect_boundary[i], :])  # hidden_size
            sw = torch.stack(_sw, dim=0)  # batch,hidden_size

        return torch.cat((s_max, sa), dim=-1)  # batch,hidden_size*2

    def forward(self, repr, aspect, len_x, aspect_boundary=None, repr2=None):
        outputs = self.output_layer(self.project(repr, aspect, len_x, aspect_boundary))
        if repr2 is not None:
            outputs += self.output_layer1(self.project(repr2, aspect, len_x, aspect_boundary))
        out = self.dense(outputs)
        return out


class Test(nn.Module):
    '''
    polar+posiiton ：自注意力 作用在h上
    word+position：gru
    smax,hn,sa 结合是最好的
        res:
            loss:0.5050 acc:80.68% f1:71.872067
        lap:
            loss:0.6600 acc:74.26% f1:70.003597
    tri,position100:
        res:
            loss:0.4929 acc:80.69% f1:71.198793
        lap:
            loss:0.6326 acc:73.28% f1:68.080303

    no-smax，sa+hn:
        res:
            loss:0.5724 acc:77.94% f1:66.004020
        lap:
            loss:0.6902 acc:73.12% f1:69.542618
    no-sa，smax + hn：sa 是位置感知
        res:
            loss:0.5090 acc:79.83% f1:69.504786
        lap:
            loss:0.6331 acc:73.98% f1:68.100453
    no-hn,s atae论文中认为加上hn会有提升
        res:
            loss:0.5128 acc:80.23% f1:70.384516
        lap:
            loss:0.6422 acc:73.79% f1:69.070650

    decoder 0.8 v0.2(encoder->tanh)
        res:
            loss:0.7196 acc:80.32% f1:70.513588 
        lap:
            loss:0.9135 acc:74.23% f1:69.399393
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        self.name = 'test'
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

    def forward(self, word, position, polar, pos, aspect, len_x):
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
        hn = encoder_out[:, -1, :]  # batch,hidden_size
        # concat
        # s = torch.cat((s_max, sp, sa), dim=-1)
        # s = torch.cat((s_max, sa, hn), dim=-1)
        s = torch.cat((s_max, hn), dim=-1)

        out = self.dense(s)
        return out, gru_out, hidden  # batch,num_polar


class Testa(nn.Module):
    '''
    类似atae 不考虑位置编码 直接作aspect的注意力计算
    输入上也不考虑 word+aspect的拼接 ，所以其实更接近at-lstm
    word：gru
    和原模型比较接近
    res:
        loss:0.5590 acc:77.77% f1:64.572226
    lap:
        loss:0.7403 acc:69.91% f1:65.607651

    smax + hn: 融合位置编码的aspect 中显示 smax最有用
                最大池化发生了作用
        res：
            loss:0.6088 acc:79.17% f1:70.223417
        lap:
            loss:0.6551 acc:73.01% f1:68.514047
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        self.name = 'testa'
        # atrribute
        self.hidden_size = hidden_dim
        # gru
        gru_input_size = word_embed_dim  # word
        self.gru = nn.GRU(gru_input_size, self.hidden_size,
                          batch_first=True, bidirectional=True)

        # decoder
        hidden_dim = hidden_dim * 2  # bidirection

        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim,
                                              score_function='bi_linear')  # aspect + context

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_polar)
        )

    def forward(self, word, position, polar, pos, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = word

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        encoder_out = gru_out
        # decoder
        # squeeze
        max_x = len_x.max()

        # sa
        hap = torch.cat((encoder_out, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size
        hn = encoder_out[:, -1, :]  # batch,hidden_size
        # concat
        # s = torch.cat((s_max, sp, sa), dim=-1)
        # s = torch.cat((s_max, sa, hn), dim=-1)
        # s = torch.cat((sa, hn), dim=-1)  # atae
        s = torch.cat((s_max, hn), dim=-1)  # atae

        out = self.dense(s)
        return out, gru_out, hidden  # batch,num_polar


class Testb(nn.Module):
    '''
    类似atae 考虑位置编码 直接作aspect的注意力计算
    只不过注意力计算的时候不考虑position

    word：gru
    no position aware;aspect+hidden做注意力计算
    sa+hn：
        res:
            loss:0.5470 acc:77.54% f1:65.686826
        lap:
            loss:0.6908 acc:71.96% f1:66.170036

    + position-aware;sa + hn;aspect+position+hidden做注意力计算
    sa+hn
        res：
            loss:0.5213 acc:79.15% f1:68.063962
        lap:
            loss:0.6579 acc:71.88% f1:66.236789

    + no-position-aware
    smax + hn 丢弃了所谓的位置感知注意力
        res:
            loss:0.4988 acc:80.71% f1:70.184447
        lap:
            loss:0.6370 acc:74.36% f1:70.390545
        对比 不考虑位置编码的模型
        smax + hn: 融合位置编码的aspect 中显示 smax最有用
                最大池化发生了作用
        res：
            loss:0.6088 acc:79.17% f1:70.223417
        lap:
            loss:0.6551 acc:73.01% f1:68.514047
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        self.name = 'testb'
        # atrribute
        self.hidden_size = hidden_dim
        # gru
        gru_input_size = word_embed_dim + position_embed_dim  # word
        self.gru = nn.GRU(gru_input_size, self.hidden_size,
                          batch_first=True, bidirectional=True)
        # decoder
        hidden_dim = hidden_dim * 2  # bidirection

        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim + position_embed_dim,
                                              score_function='bi_linear')  # aspect + context

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_polar)
        )

    def forward(self, word, position, polar, pos, aspect, len_x):
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

        # sa
        hap = torch.cat((encoder_out, position, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size
        hn = encoder_out[:, -1, :]  # batch,hidden_size
        # concat
        # s = torch.cat((s_max, sp, sa), dim=-1)
        # s = torch.cat((s_max, sa, hn), dim=-1)
        # s = torch.cat((sa, hn), dim=-1)  # atae
        s = torch.cat((s_max, hn), dim=-1)  # atae

        out = self.dense(s)
        return out, gru_out, hidden  # batch,num_polar


class Testc(nn.Module):
    '''
    类似atae 不考虑位置编码 ,输入上拼接x+aspect;
    注意力上作aspect的注意力计算
    word：gru
    和原模型比较接近
    at-lstm:
    res:
        loss:0.5590 acc:77.77% f1:64.572226
    lap:
        loss:0.7403 acc:69.91% f1:65.607651

    atae-lstm(证明在输入上拼接aspect的作用，作用不大，甚至降低了模型的性能)：
        res:
            loss:0.5584 acc:77.48% f1:65.652154
        lap:
            loss:0.7161 acc:70.17% f1:64.824529
    atae-lstm :
    smax+hn：
        res:
            loss:0.5875 acc:77.12% f1:62.206312
        lap:
            loss:0.7051 acc:70.36% f1:65.364662
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()

        self.name = 'testc'
        # atrribute
        self.hidden_size = hidden_dim
        # gru
        gru_input_size = word_embed_dim * 2  # word
        self.gru = nn.GRU(gru_input_size, self.hidden_size,
                          batch_first=True, bidirectional=True)

        # decoder
        hidden_dim = hidden_dim * 2  # bidirection

        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim,
                                              score_function='bi_linear')  # aspect + context

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_polar)
        )

    def forward(self, word, position, polar, pos, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        max_x = len_x.max()

        x = torch.cat((word, aspect.expand(-1, max_x, -1)), dim=-1)

        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)  # batch,seq_len,hidden_size*2

        encoder_out = gru_out
        # decoder
        # squeeze

        # sa
        hap = torch.cat((encoder_out, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size
        hn = encoder_out[:, -1, :]  # batch,hidden_size
        # concat
        # s = torch.cat((s_max, sp, sa), dim=-1)
        # s = torch.cat((s_max, sa, hn), dim=-1)
        # s = torch.cat((sa, hn), dim=-1)  # atae
        s = torch.cat((s_max, hn), dim=-1)  # atae

        out = self.dense(s)
        return out, gru_out, hidden  # batch,num_polar


class Test1(nn.Module):
    '''
    统一编码位置、情感词等信息送进gru，然后做位置感知注意力计算

    res:
        loss:0.5239 acc:79.50% f1:67.076283
    lap:
        loss:0.6143 acc:75.27% f1:71.287023

    + decoder 0.5 ，下降了:
        res:
            loss:0.2884 acc:77.91% f1:65.709231
        lap:
            loss:0.3559 acc:72.41% f1:67.266184
   + decoder 0.8  :
        res:
            loss:0.4207 acc:79.64% f1:67.803579
        lap:
            loss:0.5160 acc:74.17% f1:69.442229


    no-sa（x）: sa,position-aware, 是位置加权hidden
        res:
        lap:
            loss:0.6236 acc:73.95% f1:70.257347  由此可见 sa是有用的

    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()
        self.name = 'test1'
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

    def forward(self, word, position, polar, pos, aspect, len_x):
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
        return out, encoder_out, hidden  # batch,num_polar


class Test1a(nn.Module):
    '''
    统一编码位置、情感词等信息送进gru，然后做位置感知注意力计算
    位置感知时 [aspect + positon]
    res:
        loss:0.5318 acc:78.68% f1:66.104875
    lap:
        loss:0.6455 acc:73.67% f1:68.845737
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()
        self.name = 'test1a'
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

    def forward(self, word, position, polar, pos, aspect, position_zero, len_x):
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

        # sp
        # polar = torch.cat((polar,position),dim=-1)
        # polar, _ = self.attention_p(k=polar, q=polar)
        # polar = torch.tanh(polar)
        # hp = torch.cat((encoder_out, polar), dim=-1)
        # _, scores = self.attention_hp(hp)
        # sp = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # sa
        aspect_position = position_zero.expand(-1, max_x, -1)
        hap = torch.cat((encoder_out,
                         aspect.expand(-1, max_x, -1),
                         aspect_position), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size

        # concat
        s = torch.cat((s_max, sa), dim=-1)

        out = self.dense(s)
        return out, encoder_out, hidden  # batch,num_polar


class Test4(nn.Module):
    '''
    模仿imn
    polar+posiiton ：self-attention ,拼接上下文的平均向量,作为注意力计算然后加权到polar
    word_position：gru
    拼接 polar + word
    res:

    lap:
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


class Test6(nn.Module):
    '''
    多视图
    单层gru只附加位置信息:
    第二层gru 附加情感词信息:
    word+position：uni-gru
    +polar : bi-gru

        res:
            loss:0.5377 acc:77.89% f1:63.989178
        lap:
            loss:0.6270 acc:74.01% f1:69.198425

    +decoder 0.8
        res:
            loss:0.4458 acc:78.43% f1:66.596933
        lap:
            loss:0.5260 acc:74.08% f1:68.856033
            loss:0.5496 acc:72.98% f1:67.205391
    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 pos_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()
        self.name = 'test6'
        # atrribute
        self.hidden_size = hidden_dim
        # gru
        # gru_input_size = self.word_embedding_size   # word
        self.uni_gru = nn.GRU(word_embed_dim + position_embed_dim, self.hidden_size,
                              batch_first=True, bidirectional=True)
        self.bi_gru = nn.GRU(self.hidden_size + polar_embed_dim, self.hidden_size,
                             batch_first=True, bidirectional=True)

        # self.bi_gru = nn.GRU(self.hidden_size + pos_dim, self.hidden_size,
        #                      batch_first=True, bidirectional=True)  # hidden + pos

        # decoder
        hidden_dim = hidden_dim * 2  # bidirection
        self.aspect_u = nn.Linear(word_embed_dim, word_embed_dim)

        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim + position_embed_dim,
                                              score_function='bi_linear')  # aspect + context

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_polar)
        )

    def forward(self, word, position, polar, pos, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = torch.cat((word, position), dim=-1)
        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        uni_out, uni_hidden = self.uni_gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        uni_out, _ = pad_packed_sequence(uni_out, batch_first=True)  # batch,seq_len,hidden_size*2
        # uni + polar
        bi_in = uni_out[:, :, :self.hidden_size] + uni_out[:, :, self.hidden_size:]
        bi_in = torch.cat((bi_in, polar), dim=-1)  # + polar
        # bi_in = torch.cat((bi_in, pos), dim=-1)  # + pos
        bi_out, bi_hidden = self.bi_gru(bi_in)

        encoder_out = bi_out
        # decoder
        max_x = len_x.max()
        # sa
        hap = torch.cat((encoder_out, position, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size

        # concat
        s = torch.cat((s_max, sa), dim=-1)

        out = self.dense(s)
        return out, uni_out, uni_hidden  # batch,num_polar


class Test7(nn.Module):
    '''
    多视图
    单层gru   只有word:
    第二层gru 附加位置信息 position:
    通通使用smax + hn

    smax + hn：
        res:
            loss:0.5828 acc:78.17% f1:69.187666
        lap:
            loss:0.6876 acc:71.22% f1:66.234880
    + sa position-aware
    smax + hn + sa:
        res:
            loss:0.5649 acc:77.22% f1:57.798442
        lap:
            loss:0.6616 acc:71.97% f1:66.840389
    sa+hn (试图再次证明smax的作用)
        res:
            loss:0.5230 acc:78.79% f1:67.585943
        lap:
            loss:0.7332 acc:72.01% f1:66.818375s

    sa+smax:
        res:
            loss:0.5331 acc:78.93% f1:64.807596
        lap:

    采取第一层的输出（no-position）作为encoder-out
        res:
            loss:0.5588 acc:80.56% f1:71.151999
        lap:
            loss:0.7030 acc:71.88% f1:66.399327
    第一层的输出(no-position?) 修改attention，取消posiiton连接,aspect作为query
        这应该就是atae了
        res:
            loss:0.5622 acc:79.04% f1:68.539925
        lap:
            loss:0.6430 acc:74.12% f1:68.899443

    '''

    def __init__(self,
                 word_embed_dim,
                 position_embed_dim,
                 polar_embed_dim,
                 hidden_dim,
                 num_polar
                 ):
        super().__init__()
        self.name = 'test7'
        # atrribute
        self.hidden_size = hidden_dim
        # gru
        # gru_input_size = self.word_embedding_size   # word
        self.uni_gru = nn.GRU(word_embed_dim, self.hidden_size,
                              batch_first=True, bidirectional=True)
        self.bi_gru = nn.GRU(self.hidden_size + position_embed_dim,
                             self.hidden_size,
                             batch_first=True, bidirectional=True)

        # decoder
        hidden_dim = hidden_dim * 2  # bidirection

        # self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim + position_embed_dim,
        #                                       score_function='bi_linear')  # aspect + context

        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim,
                                              score_function='bi_linear')  # aspect + context

        # self.dense = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.Dropout(p=0.5),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, num_polar)
        # )

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_polar)
        )

    def forward(self, word, position, polar, pos, aspect, len_x):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = word
        max_x = len_x.max()
        # word+pos to gru
        pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        uni_out, uni_hidden = self.uni_gru(pad_x)  # hidden:layer*direction,batch,hidden_size
        uni_out, _ = pad_packed_sequence(uni_out, batch_first=True)  # batch,seq_len,hidden_size*2
        # uni + position
        bi_in = uni_out[:, :, :self.hidden_size] + uni_out[:, :, self.hidden_size:]
        bi_in = torch.cat((bi_in, position), dim=-1)  # + polar
        bi_out, bi_hidden = self.bi_gru(bi_in)

        encoder_out = bi_out  # 第二层的输出
        encoder_out = uni_out  # 第一次的输出
        # decoder
        # sa
        # hap = torch.cat((encoder_out, position, aspect.expand(-1, max_x, -1)), dim=-1)
        hap = torch.cat((encoder_out, aspect.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        # smax
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size
        hn = encoder_out[:, -1, :]  # batch,hidden_size
        # concat
        # s = torch.cat((s_max, sa,hn), dim=-1)
        s = torch.cat((s_max, sa), dim=-1)

        out = self.dense(s)
        return out, uni_out, uni_hidden  # batch,num_polar
