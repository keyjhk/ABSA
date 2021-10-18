import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from data_utils import SOS_TOKEN
from layers.attention import Attention, NoQueryAttention, squeeze_embedding


class Decoder(nn.Module):
    def __init__(self,
                 num_polar,
                 word_embedding,
                 polar_embedding,
                 hidden_size,
                 tokenizer,
                 force_teach_ratio=0.5,
                 device='cpu',
                 mode='aesc'  # aesc,alsc
                 ):
        super().__init__()

        # attribute
        self.num_polar = num_polar
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.force_teach_ratio = force_teach_ratio
        self.device = device
        self.mode = mode
        # embedding
        self.word_embedding = word_embedding
        self.polar_embedding = polar_embedding
        # gru
        self.decoder = nn.GRU(self.word_embedding.embedding_dim, self.hidden_size, batch_first=True,
                              num_layers=2, dropout=0.2)

    def mask_loss(self, predict, target, mask, max_x, polar_flag):
        # predict:batch,seq_len+num_polar; target:batch;
        # mask:batch ; the not pad_token is True

        # need to format target, polar_idx + len_max
        if polar_flag == 1:
            _target = target + max_x
        else:
            _target = target
        loss = -torch.log(torch.gather(predict, 1, _target.view(-1, 1)).squeeze(1))  # batch
        loss = loss.masked_select(mask).mean()  # batch的平均损失
        return loss

    def index_convert(self, idxs, x, max_x):
        # convert idx(position in sentence indices) to token_idx/polar/idx ,then to their embeddings respectively
        # idxs:batch ; x:batch,max_len ;max_x :batch,max seq_len in batch sentences
        token_polar_idx = []
        for i in range(len(idxs)):  # batch_size
            _idx = idxs[i]
            _x = x[i]  # max_len
            if _idx < max_x:  # token_position
                token_idx = _x[_idx]
                token_polar_idx.append(self.word_embedding(token_idx))
            else:
                polar_idx = _idx - max_x
                token_polar_idx.append(self.polar_embedding(polar_idx))

        return torch.stack(token_polar_idx).unsqueeze(dim=1)  # batch,1,embed_size

    def calculate_loss(self, x, len_x, encoder_out, hidden, target, len_t, mask_t):
        # x(means context):batch,max_len ;encoder_out:batch,seq_len,hidden_size
        # hidden:layers*2,batch,hidden_size ; target: batch,max_len ; lent:batch
        # mask_t: batch,max_len

        max_x = len_x.max()
        max_t = len_t.max()

        out = self.predict(x, len_x, encoder_out, hidden, target, len_t)  # max_t,batch,seq_len+num+polar
        loss = 0
        polar_flag = self.build_polar_flag(max_t)
        for t in range(max_t):
            if self.mode == 'aesc':
                loss += self.mask_loss(out[t], target[:, t], mask_t[:, t], max_x, polar_flag[t])
            elif self.mode == 'alsc':
                if polar_flag[t] == 1:
                    loss += self.mask_loss(out[t], target[:, t], mask_t[:, t], max_x, polar_flag[t])

        return loss / max_t

    def build_polar_matrix(self):
        polar_labels = torch.tensor(range(self.num_polar)).to(self.device)
        polar_matrix = self.polar_embedding(polar_labels)  # num_polar,polar_embed_size

        return polar_matrix

    def build_polar_flag(self, max_t):
        # mark polar idx. 001001...
        polar_flag = torch.zeros(max_t)  # max_t
        for i in range(2, max_t, 3):
            polar_flag[i] = 1  # this position is polar
        return polar_flag

    def get_aspect_embedding(self, x, aspect_positions):
        # batch,num_aspect
        aspects = []
        num_aspect = aspect_positions.shape[1]
        for _batch in range(aspect_positions.shapep[0]):
            for i in range(0, len(num_aspect), 2):
                s, e = aspect_positions[_batch, i], aspect_positions[_batch, i + 1]
                aspect = self.word_embedding(x[_batch, s:e + 1])
                aspects.append(aspect / (e - s + 1))
        return torch.stack(aspects)  # ?,word_embed_size

    def predict(self, x, len_x, encoder_out, hidden, target, len_t):
        # x(means context):batch,max_len ;encoder_out:batch,seq_len,hidden_size
        # hidden:layers*2,batch,hidden_size ; target: batch,max_len ; lent:batch
        # mask_t: batch,max_len

        max_t = len_t.max()
        max_x = len_x.max()
        mb_size = encoder_out.shape[0]

        polar_flag = self.build_polar_flag(max_t)  # max_t,001001...
        polar_matrix = self.build_polar_matrix()  # num_polar,polar_embed_size

        sos_tokens = torch.tensor([self.tokenizer.word2idx[SOS_TOKEN]] * mb_size).view(-1, 1).to(self.device)  # batch,1
        decoder_in = self.word_embedding(sos_tokens)  # batch,1,embed_size
        last_hidden = hidden[-2:]  # last_layer's forward/backward

        output = []
        for t in range(max_t):
            # decoder_out: batch,1,hidden_size ;last_hidden:layers,batch,hidden_size
            decoder_out, last_hidden = self.decoder(decoder_in, last_hidden)

            # classify
            hp = torch.cat((encoder_out, polar_matrix.expand(mb_size, -1, -1)),
                           dim=1)  # batch,seq_len+num_polar,hidden_size
            scores = torch.bmm(hp, decoder_out.transpose(1, 2)).squeeze(-1)  # batch,seq_len+num_pos
            scores = F.softmax(scores, dim=-1)  # batch,seq_len+num_pos ; probability of tokens_position/polar_idx
            idxs = torch.argmax(scores, dim=-1)  # batch
            output.append(scores)

            force_teach = random.random() > self.force_teach_ratio

            if self.mode == 'aesc':  # extract aspect boundary,and classify sentiment polarity
                if not force_teach:
                    decoder_in = self.index_convert(idxs, x, max_x)  # batch,1,embed_size
                else:
                    if polar_flag[t] == 1:
                        decoder_in = self.index_convert(target[:, t] + max_x, x, max_x)
                    else:
                        decoder_in = self.index_convert(target[:, t], x, max_x)
                output.append(scores)
            elif self.mode == 'alsc':  # aspect boundary is given,just classify sentiment polarity
                if polar_flag[t] == 1:  # mode only predicts the polarity
                    if not force_teach:
                        decoder_in = self.index_convert(idxs, x, max_x)
                    else:
                        decoder_in = self.index_convert(target[:, t] + max_x, x, max_x)
                else:
                    decoder_in = self.index_convert(target[:, t], x, max_x)  # given the aspect boundary

        return torch.stack(output)  # max_t,batch,seq_len+num_polar


class Primay(nn.Module):
    def __init__(self, hidden_dim, word_embed_dim, polar_dim, num_polar):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.polar_gru = nn.GRU(polar_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

        self.aspect_u = nn.Linear(word_embed_dim, word_embed_dim)
        self.attention_pc = NoQueryAttention(hidden_dim * 2,
                                             score_function='bi_linear')  # polar_hidden and context
        self.attention_ac = NoQueryAttention(word_embed_dim + hidden_dim,
                                             score_function='bi_linear')  # aspect and context
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_polar),
        )

    def forward(self, encoder_out, polar, aspect, len_x):
        # encoder_out:batch,seq_len,hidden_size ;polar:batch,seq_len,polar_size;
        # aspect:batch,1,word_embed_dim

        max_x = len_x.max()
        # squeeze embedding
        pad_polar = pack_padded_sequence(polar, len_x.cpu(),
                                         batch_first=True, enforce_sorted=False)
        polar_out, _ = self.polar_gru(pad_polar)
        polar_out, _ = pad_packed_sequence(polar_out, batch_first=True)  # batch,seq_len,polar_dim
        polar_out = torch.mean(polar_out, dim=1).unsqueeze(1)  # batch,1,hidden_dim

        pc = torch.cat((encoder_out, polar_out.expand(-1, max_x, -1)), dim=-1)  # batch,seq_len,hidden_dim*2
        _, scores = self.attention_pc(pc)  # batch,1,seq_len
        r_pc = torch.bmm(scores, encoder_out)  # batch,1,hidden_size

        # aspect_u = self.aspect_u(aspect)  # batch,1,word_embed_dim
        aspect_u = aspect  # batch,1,word_embed_dim
        ac = torch.cat((encoder_out, aspect_u.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_ac(ac)
        r_ac = torch.bmm(scores, encoder_out)  # batch,1,hidden_size

        r = torch.cat((r_pc, r_ac), dim=-1).squeeze(1)  # batch,hidden_size*2

        out = self.dense(r)  # batch,num_polar  交叉熵会进行softmax
        return out


class PrimayPolarSA(nn.Module):
    def __init__(self, hidden_dim, word_embed_dim, polar_dim, num_polar):
        super().__init__()
        assert hidden_dim % 2 == 0

        self.aspect_u = nn.Linear(word_embed_dim, word_embed_dim)
        self.sa = Attention(hidden_dim)  # hidden_dim
        self.attention_pc = NoQueryAttention(polar_dim + hidden_dim,
                                             score_function='bi_linear')  # polar + context
        self.attention_ac = NoQueryAttention(word_embed_dim + hidden_dim,
                                             score_function='bi_linear')  # aspect + context
        self.mhsa = Attention(hidden_dim * 3, n_head=5)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 3, num_polar)
        )

    def forward(self, encoder_out, polar, aspect, len_x):
        # encoder_out:batch,seq_len,hidden_size ;polar:batch,seq_len,polar_size;
        # aspect:batch,1,word_embed_dim

        max_x = len_x.max()
        # squeeze embedding
        polar_out = squeeze_embedding(polar, len_x.cpu())  # batch,seq_len,polar_dim

        pc = torch.cat((encoder_out, polar_out), dim=-1)  # batch,seq_len,hidden_dim+polar_dim
        _, scores = self.attention_pc(pc)  # batch,1,seq_len
        r_pc = torch.bmm(scores, encoder_out)  # batch,1,hidden_size

        # aspect_u = self.aspect_u(aspect)  # batch,1,word_embed_dim
        aspect_u = aspect
        ac = torch.cat((encoder_out, aspect_u.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_ac(ac)
        r_ac = torch.bmm(scores, encoder_out)  # batch,1,hidden_size

        smax = torch.max(encoder_out, dim=1)[0].unsqueeze(1)
        r = torch.cat((r_pc, r_ac, smax), dim=-1).squeeze(1)  # batch,hidden_size*2

        r, _ = self.mhsa(k=r, q=r)

        out = self.dense(r)  # batch,num_polar  交叉熵会进行softmax
        return out


class PrimayNOAu(nn.Module):
    def __init__(self, hidden_dim, word_embed_dim, polar_dim, num_polar):
        super().__init__()
        assert hidden_dim % 2 == 0

        self.aspect_u = nn.Linear(word_embed_dim, word_embed_dim)
        self.sa = Attention(hidden_dim)  # hidden_dim
        self.attention_pc = NoQueryAttention(polar_dim + hidden_dim,
                                             score_function='bi_linear')  # polar + context
        self.attention_ac = NoQueryAttention(word_embed_dim + hidden_dim,
                                             score_function='bi_linear')  # aspect + context
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_polar)
        )

    def forward(self, encoder_out, polar, aspect, len_x):
        # encoder_out:batch,seq_len,hidden_size ;polar:batch,seq_len,polar_size;
        # aspect:batch,1,word_embed_dim

        max_x = len_x.max()
        # squeeze embedding
        polar_out = squeeze_embedding(polar, len_x.cpu())  # batch,seq_len,polar_dim
        # self attention here
        _, scores = self.sa(q=encoder_out, k=encoder_out)  # batch,seq_len,seq_len
        polar_out = torch.bmm(scores, polar_out)  # batch,seq_len,polar_embed_dim

        pc = torch.cat((encoder_out, polar_out), dim=-1)  # batch,seq_len,hidden_dim+polar_dim
        _, scores = self.attention_pc(pc)  # batch,1,seq_len
        r_pc = torch.bmm(scores, encoder_out)  # batch,1,hidden_size

        # aspect_u = self.aspect_u(aspect)  # batch,1,word_embed_dim
        aspect_u = aspect  # batch,1,word_embed_dim
        ac = torch.cat((encoder_out, aspect_u.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_ac(ac)
        r_ac = torch.bmm(scores, encoder_out)  # batch,1,hidden_size

        r = torch.cat((r_pc, r_ac), dim=-1).squeeze(1)  # batch,hidden_size*2

        out = self.dense(r)  # batch,num_polar  交叉熵会进行softmax
        return out


class PrimayATAE(nn.Module):
    def __init__(self, hidden_dim, word_embed_dim, polar_dim, num_polar):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.aspect_u = nn.Linear(word_embed_dim, word_embed_dim)
        self.attention_ac = NoQueryAttention(word_embed_dim + hidden_dim,
                                             score_function='bi_linear')  # aspect + context
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, num_polar)
        )

    def forward(self, encoder_out, polar, aspect, len_x, aspectu=False):
        # encoder_out:batch,seq_len,hidden_size ;polar:batch,seq_len,polar_size;
        # aspect:batch,1,word_embed_dim

        max_x = len_x.max()
        # squeeze embedding

        # aspect_u = self.aspect_u(aspect)  # batch,1,word_embed_dim
        if aspectu == False:
            aspect_u = aspect
        else:
            aspect_u = self.aspect_u(aspect)  # batch,1,word_embed_dim

        ac = torch.cat((encoder_out, aspect_u.expand(-1, max_x, -1)), dim=-1)
        _, scores = self.attention_ac(ac)
        r_ac = torch.bmm(scores, encoder_out)  # batch,1,hidden_size

        # atae just use r_ac
        out = self.dense(r_ac)  # batch,1,num_polar  交叉熵会进行softmax
        return out.squeeze(1)  # batch,num_polar


class PrimayPosition(nn.Module):
    def __init__(self, hidden_dim, word_embed_dim, position_dim, num_polar):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.aspect_u = nn.Linear(word_embed_dim, word_embed_dim)
        self.attention_p = Attention(50)
        self.attention_hp = NoQueryAttention(50 + hidden_dim, score_function='bi_linear')
        self.attention_hap = NoQueryAttention(word_embed_dim + hidden_dim + position_dim,
                                              score_function='bi_linear')  # aspect + context
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_polar),

        )

    def forward(self, encoder_out, position, polar, aspect, len_x):
        # encoder_out:batch,seq_len,hidden_size ;polar:batch,seq_len,polar_size;
        # aspect:batch,1,word_embed_dim

        max_x = len_x.max()

        # aspect_u = self.aspect_u(aspect)  # batch,1,word_embed_dim
        aspect_u = aspect

        position = squeeze_embedding(position, len_x.cpu())
        polar = squeeze_embedding(polar, len_x.cpu())
        polar, _ = self.attention_p(k=polar, q=polar)

        hp = torch.cat((encoder_out, polar), dim=-1)
        _, scores = self.attention_hp(hp)
        sp = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size

        hap = torch.cat((encoder_out, position, aspect_u.expand(-1, max_x, -1)), dim=-1)

        _, scores = self.attention_hap(hap)
        sa = torch.bmm(scores, encoder_out).squeeze(1)  # batch,hidden_size
        s_max = torch.max(encoder_out, dim=1)[0].squeeze(1)  # batch,hidden_size
        hn = encoder_out[:, -1, :]  # batch,hidden_size

        s = torch.cat((sa, s_max, hn), dim=-1)  # batch,hidden_size*3

        # test
        # s = torch.cat((hn, s_max), dim=-1)  # batch,hidden_size*2

        # atae just use r_ac
        out = self.dense(s)
        return out  # batch,num_polar
