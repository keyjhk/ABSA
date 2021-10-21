import torch
import torch.nn as nn
from models.encoder import *
from models.primary import *
from models import atae_lstm


class CVTModel(nn.Module):
    def __init__(self,
                 num_pos,
                 num_polar,
                 num_position,
                 pretrained_embedding,
                 tokenizer,
                 mode='labeled',
                 ### parameters for model ###
                 word_embedding_size=300,
                 pos_embedding_size=50,
                 polar_embedding_size=50,
                 position_embedding_size=50,
                 encoder_hidden_size=300,
                 ):

        super().__init__()

        # inputs cols: 和 forward所传参数顺序一致
        self.inputs_cols = [
            'context_indices', 'pos_indices', 'polar_indices',
            'text_indices', 'position_indices',
            'aspect_indices', 'aspect_boundary',
            'target',
            'len_s',
        ]

        self.mode = mode
        self.tokenizer = tokenizer
        # embedding for word/pos/polar
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.tensor(pretrained_embedding, dtype=torch.float))
        self.pos_embedding = nn.Embedding(num_pos, pos_embedding_size)
        self.polar_embedding = nn.Embedding(num_polar, polar_embedding_size)
        self.position_embedding = nn.Embedding(num_position + 1, position_embedding_size)

        # encoder
        self.encoder = Test1(word_embed_dim=word_embedding_size,
                             position_embed_dim=position_embedding_size,
                             polar_embed_dim=polar_embedding_size,
                             hidden_dim=encoder_hidden_size,
                             num_polar=num_polar)
        # location decoder
        self.location_decoder = LocationDecoder(self.word_embedding,
                                                encoder_hidden_size,
                                                tokenizer=tokenizer)

        self.name = self.encoder.name

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

        # meta data
        context = inputs[self.inputs_cols.index('context_indices')]  # batch,MAX_LEN
        pos = inputs[self.inputs_cols.index('pos_indices')]  # batch,MAX_LEN
        polar = inputs[self.inputs_cols.index('polar_indices')]  # batch,MAX_LEN
        text_indices = inputs[self.inputs_cols.index('text_indices')]  # batch,MAX_LEN
        position_indices = inputs[self.inputs_cols.index('position_indices')]  # batch,MAX_LEN
        aspect_indices = inputs[self.inputs_cols.index('aspect_indices')]  # batch,MAX_LEN
        aspect_boundary = inputs[self.inputs_cols.index('aspect_boundary')]  # batch,2
        target = inputs[self.inputs_cols.index('target')]  # batch
        len_x = inputs[self.inputs_cols.index('len_s')]  # batch

        # word/polar/pos/position # batch,MAX_LEN,word_embed_dim
        word = self.word_embedding(context)  # batch,MAX_LEN,word_embed_dim
        pos = self.pos_embedding(pos)  # batch,MAX_LEN,pos_embed_dim
        polar = self.polar_embedding(polar)  # batch,MAX_LEN,polar_embed_dim
        position = self.position_embedding(position_indices)  # batch,MAX_LEN,position_embed_dim
        # squeeze
        word = squeeze_embedding(word, len_x)
        pos = squeeze_embedding(pos, len_x)
        polar = squeeze_embedding(polar, len_x)
        position = squeeze_embedding(position, len_x)

        # pool(average) aspect
        aspect_pool = self.pool_aspect(aspect_indices, aspect_boundary)

        loss = 0
        if mode == "labeled":  # 监督训练
            # self._unfreeze_model()
            out,encoder_out,last_hidden = self.encoder(word, position, polar, aspect_pool, len_x)
            loss += self.loss(out, target)
            # location loss
            locations = self.location_decoder(encoder_out,
                                              last_hidden,
                                              context) # batch,2,seq_len
            loss += (self.loss(locations[:,0,:],aspect_boundary[:,0]) +
                     self.loss(locations[:, 1, :], aspect_boundary[:, 1])
                     )
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
