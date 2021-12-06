import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from layers.attention import Attention, squeeze_embedding, NoQueryAttention
from data_utils import SOS_TOKEN


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, batch_first=False, num_layers=1):
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size,
                          bidirectional=bidirectional, batch_first=batch_first, num_layers=num_layers)

    def forward(self, x, len_x=None, hidden=None):
        self.gru.flatten_parameters()
        if len_x is not None:
            pad_x = pack_padded_sequence(x, len_x.cpu(), batch_first=self.batch_first, enforce_sorted=False)
            gru_out, gru_hidden = self.gru(pad_x, hidden)  # hidden:layer*direction,batch,hidden_size
            gru_out, _ = pad_packed_sequence(gru_out, batch_first=self.batch_first)
        else:
            gru_out, gru_hidden = self.gru(x, hidden)
        return gru_out, gru_hidden


class PolarDecoder(nn.Module):
    def __init__(self,
                 word_embedding,
                 hidden_size,
                 num_polar,
                 tokenizer,
                 name="primary", p=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.word_embedding = word_embedding
        self.num_polar = num_polar
        self.name = name

        word_dim = word_embedding.embedding_dim
        self.decoder = nn.GRU(word_dim,
                              hidden_size, batch_first=True)

        self.attention_ch = Attention(hidden_size)  # hidden/context score_function='mlp'
        self.attention_dropout = nn.Dropout(p=p)
        self.dense = nn.Linear(hidden_size * 3, num_polar)

    def clone(self, name, p=None):
        # share decoder /word embed
        # attention mechanism/dense  is different
        p = p if p else self.attention_dropout.p
        instance = self.__class__(word_embedding=self.word_embedding,
                                  hidden_size=self.hidden_size,
                                  num_polar=self.num_polar,
                                  tokenizer=self.tokenizer,
                                  p=p,
                                  name=name,
                                  )
        instance.decoder = self.decoder
        return instance

    def export_max(self, encoder_out):
        # export max for vision,called when debug
        import pickle
        max_indice = torch.max(encoder_out, dim=1, keepdim=True)[1]
        pickle.dump(max_indice.cpu().numpy(), open('max_indice.pkl', 'wb'))

    def forward(self, encoder_out, last_hidden, mask_attention=None):
        # encoder: batch,seq_len,hidden_size ;
        # aspect : # batch,1,embed_dim
        # last_hidden: 2(directions),batch,hidden_size
        mb, seq = encoder_out.shape[0], encoder_out.shape[1]
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
        self.decoder.flatten_parameters()
        decoder_out, _ = self.decoder(decoder_out, last_hidden)
        _, score = self.attention_ch(q=decoder_out, k=encoder_out, mask=mask_attention)  # batch,1,seq_len
        score = self.attention_dropout(score)
        context = torch.bmm(score, encoder_out)  # batch,1,hidden_size

        smax = torch.max(encoder_out, dim=1, keepdim=True)[0]  # batch,1,hidden_size
        ch = torch.cat((context, decoder_out, smax), dim=-1)  # batch,1,hidden_size*3
        ch = ch.squeeze(1)  # batch,hidden_size*3

        out = self.dense(ch)  # batch,num_polar

        return out


class RAMDecoder(nn.Module):
    def __init__(self,
                 word_embedding,
                 hidden_size,
                 num_polar,
                 tokenizer,
                 name="primary", p=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.word_embedding = word_embedding
        self.num_polar = num_polar
        self.name = name

        word_dim = word_embedding.embedding_dim
        self.decoder = nn.GRU(word_dim,
                              hidden_size, batch_first=True)

        self.attention_ch = Attention(hidden_size)  # hidden/context score_function='mlp'
        self.attention_dropout = nn.Dropout(p=p)
        self.dense = nn.Linear(hidden_size * 2, num_polar)

    def clone(self, name, p=None):
        # share decoder /word embed
        # attention mechanism/dense  is different
        p = p if p else self.attention_dropout.p
        instance = self.__class__(word_embedding=self.word_embedding,
                                  hidden_size=self.hidden_size,
                                  num_polar=self.num_polar,
                                  tokenizer=self.tokenizer,
                                  p=p,
                                  name=name,
                                  )
        instance.decoder = self.decoder
        return instance

    def forward(self, encoder_out, last_hidden, mask_attention=None):
        # encoder: batch,seq_len,hidden_size ;
        # aspect : # batch,1,embed_dim
        # last_hidden: 2(directions),batch,hidden_size
        mb, seq = encoder_out.shape[0], encoder_out.shape[1]
        device = encoder_out.device

        # smax as  decoder input
        decoder_out = torch.max(encoder_out, dim=1, keepdim=True)[0]  # batch,1,hidden_size

        # int decoder hidden
        # last_hidden from encoder, merge forward/backward
        last_hidden = last_hidden[0] + last_hidden[1]  # batch,hidden
        last_hidden = last_hidden.unsqueeze(0)  # 1, batch,hidden_size

        # decoder
        # decoder_out: batch,1,hidden_size  ; encoder: batch,seq,hidden_size
        self.decoder.flatten_parameters()
        decoder_out, _ = self.decoder(decoder_out, last_hidden)
        _, score = self.attention_ch(q=decoder_out, k=encoder_out, mask=mask_attention)  # batch,1,seq_len
        score = self.attention_dropout(score)
        context = torch.bmm(score, encoder_out)  # batch,1,hidden_size

        # smax = torch.max(encoder_out, dim=1, keepdim=True)[0]  # batch,1,hidden_size
        ch = torch.cat((context, decoder_out), dim=-1)  # batch,1,hidden_size*3
        ch = ch.squeeze(1)  # batch,hidden_size*3

        out = self.dense(ch)  # batch,num_polar

        return out


class PositionEncoder(nn.Module):

    def __init__(self, opt):
        super().__init__()
        # atrribute
        self.uni_gru = DynamicGRU(opt.word_embedding_size + opt.position_embedding_size,
                                  opt.encoder_hidden_size,
                                  batch_first=True, bidirectional=True)
        self.uni_dropout_lab = nn.Dropout(p=opt.drop_lab)
        self.uni_dropout_unlab = nn.Dropout(p=opt.drop_unlab)

    def forward(self, word, position, len_x, mode):
        # word/pos/polar: batch,MAX_LEN,embedding_size  ;len_x:batch
        x = torch.cat((word, position), dim=-1)
        uni_out, uni_hidden = self.uni_gru(x, len_x.cpu())
        if mode == 'labeled':
            uni_out = self.uni_dropout_lab(uni_out)
        else:
            uni_out = self.uni_dropout_unlab(uni_out)

        return uni_out, uni_hidden  # for decoder
