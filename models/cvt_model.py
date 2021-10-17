import torch
import torch.nn as nn
from models.encoder import *
from models.primary import *
from models import atae_lstm


class CVTModel(nn.Module):
    def __init__(self,
                 num_pos,
                 num_polar,
                 pretrained_embedding,
                 mode='labeled',
                 ### parameters for model ###
                 word_embedding_size=300,
                 pos_embedding_size=50,
                 polar_embedding_size=300,
                 encoder_hidden_size=300,
                 combine='cvt'
                 ):

        super().__init__()

        # model name
        self.name = combine
        # inputs cols: 和 forward所传参数顺序一致
        self.inputs_cols = [
            'context_indices', 'pos_indices', 'polar_indices', 'text_indices',
            'aspect_indices', 'aspect_boundary',
            'target',
            'len_s',
        ]

        self.mode = mode
        # embedding for word/pos/polar
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.tensor(pretrained_embedding, dtype=torch.float))
        self.pos_embedding = nn.Embedding(num_pos, pos_embedding_size)
        self.polar_embedding = nn.Embedding(num_polar, polar_embedding_size)

        if self.name == 'cvt':
            self.encoder = Encoder(self.word_embedding,
                                   self.pos_embedding,
                                   hidden_size=encoder_hidden_size)
            self.primary = Primay(hidden_dim=encoder_hidden_size * 2,
                                  word_embed_dim=word_embedding_size,
                                  polar_dim=polar_embedding_size,
                                  num_polar=num_polar)
        elif self.name == 'cvte1':
            # encoder layer 1
            self.encoder = Encoderlayer1(self.word_embedding,
                                         self.pos_embedding,
                                         hidden_size=encoder_hidden_size)
            self.primary = Primay(hidden_dim=encoder_hidden_size * 2,
                                  word_embed_dim=word_embedding_size,
                                  polar_dim=polar_embedding_size,
                                  num_polar=num_polar)
        elif self.name == 'cvtnopos':
            # don't encode pos
            self.encoder = EncoderNoPOS(self.word_embedding,
                                        self.pos_embedding,
                                        hidden_size=encoder_hidden_size)
            self.primary = Primay(hidden_dim=encoder_hidden_size * 2,
                                  word_embed_dim=word_embedding_size,
                                  polar_dim=polar_embedding_size,
                                  num_polar=num_polar)
        elif self.name == 'cvtnopos-polarsa':
            # don't encode pos
            self.encoder = EncoderNoPOS(self.word_embedding,
                                        self.pos_embedding,
                                        hidden_size=encoder_hidden_size)
            self.primary = PrimayPolarSA(hidden_dim=encoder_hidden_size * 2,
                                         word_embed_dim=word_embedding_size,
                                         polar_dim=polar_embedding_size,
                                         num_polar=num_polar)
        elif self.name == 'cvt-polarsa':
            # don't encode pos
            self.encoder = Encoder(self.word_embedding,
                                        self.pos_embedding,
                                        hidden_size=encoder_hidden_size)
            self.primary = PrimayPolarSA(hidden_dim=encoder_hidden_size * 2,
                                         word_embed_dim=word_embedding_size,
                                         polar_dim=polar_embedding_size,
                                         num_polar=num_polar)
        elif self.name == 'cvt-polarnoau':
            # don't encode pos
            self.encoder = Encoder(self.word_embedding,
                                        self.pos_embedding,
                                        hidden_size=encoder_hidden_size)
            self.primary = PrimayNOAu(hidden_dim=encoder_hidden_size * 2,
                                         word_embed_dim=word_embedding_size,
                                         polar_dim=polar_embedding_size,
                                         num_polar=num_polar)


        elif self.name == 'atae':
            self.encoder = atae_lstm.ATAE_LSTM(num_polar, self.word_embedding, encoder_hidden_size)
            self.primary = None

        # loss
        self.loss = nn.CrossEntropyLoss()

    def pool_aspect(self, aspect_indices, aspect_boundary):
        aspect_len = aspect_boundary[:, 1] - aspect_boundary[:, 0] + 1  # batch
        aspect = self.word_embedding(aspect_indices)  # batch,MAX_LEN,word_embed_dim
        aspect_pool = torch.div(torch.sum(aspect, dim=1),
                                aspect_len.unsqueeze(1)).unsqueeze(1)  # batch,1,embed_dim

        return aspect_pool

    def forward(self, *inputs,
                mode='labeled'):

        context = inputs[self.inputs_cols.index('context_indices')]  # batch,MAX_LEN
        pos = inputs[self.inputs_cols.index('pos_indices')]  # batch,MAX_LEN
        polar = inputs[self.inputs_cols.index('polar_indices')]  # batch,MAX_LEN
        text_indices = inputs[self.inputs_cols.index('text_indices')]  # batch,MAX_LEN
        aspect_indices = inputs[self.inputs_cols.index('aspect_indices')]  # batch,MAX_LEN
        aspect_boundary = inputs[self.inputs_cols.index('aspect_boundary')]  # batch,2
        target = inputs[self.inputs_cols.index('target')]  # batch
        len_x = inputs[self.inputs_cols.index('len_s')]  # batch

        # pool(average) aspect
        aspect_pool = self.pool_aspect(aspect_indices, aspect_boundary)
        # polar_embedding
        polar = self.polar_embedding(polar)  # batch,MAX_LEN,word_embed_dim

        if mode == "labeled":  # 监督训练
            # self._unfreeze_model()
            if self.name.startswith('cvt'):
                # encoder_out:batch,seq_len,hidden_size*2
                encoder_out = self.encoder(context, pos, len_x)  # get encoder out
                out = self.primary(encoder_out, polar, aspect_pool, len_x)
            elif self.name == 'atae':
                out = self.encoder(context, aspect_pool, len_x)  # resume atae

            loss = self.loss(out, target)
            return loss, out
        elif mode == "unlabeled":  # 无监督训练
            pass
            # self._freeze_model()
            # # calculate the prediction and loss of the  auxiliary modules
            # # loss += loss_full + loss_forwards + loss_backwards + loss_future + loss_past
            # torch.cuda.empty_cache()  # 为什么要empty
            # return loss

    def _freeze_model(self):
        # freeze primary only; encoder is unfreezed
        self.primary.eval()
        for params in self.primary.parameters():
            params.requires_grad = False

    def _unfreeze_model(self):
        self.primary.train()
        for params in self.primary.parameters():
            params.requires_grad = True
