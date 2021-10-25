import torch
import torch.nn as nn
import torch.nn.functional as F
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
                 position_embedding_size=50,  # 50 100 300
                 encoder_hidden_size=200  # 150 200 300
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
        self.alpha = 1  # 0.5 0.8 1
        self.tokenizer = tokenizer
        # embedding for word/pos/polar
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.tensor(pretrained_embedding, dtype=torch.float))
        self.pos_embedding = nn.Embedding(num_pos, pos_embedding_size)
        self.polar_embedding = nn.Embedding(num_polar, polar_embedding_size)
        self.position_embedding = nn.Embedding(num_position + 1, position_embedding_size)

        # encoder
        # self.encoder = EncoderPosition(word_embed_dim=word_embedding_size,
        #                                position_embed_dim=position_embedding_size,
        #                                hidden_size=encoder_hidden_size,
        #                                )
        self.encoder = BilayerEncoder(word_embed_dim=word_embedding_size,
                                      position_embed_dim=position_embedding_size,
                                      hidden_size=encoder_hidden_size,
                                      )

        # test
        # self.encoder = Test6(word_embed_dim=word_embedding_size,
        #                     position_embed_dim=position_embedding_size,
        #                     polar_embed_dim=polar_embedding_size,
        #                      pos_dim=pos_embedding_size,
        #                     hidden_dim=encoder_hidden_size,
        #                     num_polar=num_polar)

        # primary
        self.primary = BilayerPrimary(word_embed_dim=word_embedding_size,
                                      hidden_dim=encoder_hidden_size,
                                      num_polar=num_polar)
        # auxiliary
        # full
        self.primary_full = BilayerPrimary(word_embed_dim=word_embedding_size,
                                      hidden_dim=encoder_hidden_size,
                                      num_polar=num_polar)

        # uni
        self.primary_uni = BilayerPrimary(word_embed_dim=word_embedding_size,
                                          hidden_dim=encoder_hidden_size,
                                          num_polar=num_polar)
        # mask
        self.primary_mask = BilayerPrimary(word_embed_dim=word_embedding_size,
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

    def dynamic_mask(self, features, position_indices, len_x, threshold=4, ratio=1):
        # features: batch,seq_len,hidden_size
        # position: batch,MAX_LEN  ; len_x: batch
        # a vision radium of a sentence
        mb = features.shape[0]
        device = position_indices.device
        window_size = torch.div(len_x, 4, rounding_mode='floor').unsqueeze(1)  # batch,1 ;half of a sentence
        threshold = torch.tensor([threshold] * mb, device=device).unsqueeze(1)  # batch,1 4
        window_size = torch.cat((window_size, threshold), dim=-1)  # batch,2
        window_size = window_size.min(dim=-1, keepdim=True)[0]  # batch,1

        max_x = len_x.max()
        # mask over threshoulds
        mask_r = torch.rand(position_indices.shape, device=device)  # batch,MAX_LEN
        mask_r = mask_r.masked_fill(mask=mask_r < ratio, value=0)  # 随机置0
        mask_r = mask_r.masked_fill(mask=mask_r >= ratio, value=1)  # 其余置1
        mask_r = mask_r.masked_fill(position_indices <= window_size, value=1)  # aspect
        mask_r = mask_r[:, :max_x].unsqueeze(dim=2)

        return features * mask_r  # batch,MAX_LEN,1

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

        # uni_out,bi_out from encoder
        # encoder_out, _ = self.encoder(word, position, len_x)
        encoder_out, uni_out = self.encoder(word, position, len_x)
        loss = 0
        if mode == "labeled":  # 监督训练
            self._unfreeze_model()
            # loss += self.alpha * self.loss(out, target)
            # location loss
            # locations = self.location_decoder(encoder_out,
            #                                   last_hidden,
            #                                   context)  # batch,2,seq_len
            # loss2 = (1 - self.alpha) * (self.loss(locations[:, 0, :], aspect_boundary[:, 0]) +
            #                             self.loss(locations[:, 1, :], aspect_boundary[:, 1])
            #                             )

            # out = self.primary(bi_out,aspect_pool,len_x)
            # encoder_out = self.dynamic_mask(encoder_out, position_indices, len_x)  # batch,seq_len,hidden_size

            out = self.primary(encoder_out, aspect_pool, len_x, repr_2=uni_out)
            loss += self.loss(out, target)
            return loss, out
        elif mode == "unlabeled":  # 无监督训练
            self._freeze_model()
            label_primary = self.primary(encoder_out, aspect_pool, len_x,repr_2=uni_out)  # batch,num_polar
            label_primary = label_primary.detach()
            # auxiliary1
            # full
            out_full = self.primary_full(encoder_out, aspect_pool, len_x, repr_2=uni_out)
            loss_full = self.loss(out_full,label_primary.argmax(dim=1))
            # masked encoder
            mask_e = self.dynamic_mask(encoder_out, position_indices, len_x)  # batch,seq_len,hidden_size
            mask_u = self.dynamic_mask(uni_out, position_indices, len_x)
            out_mask = self.primary_mask(mask_e, aspect_pool, len_x)  # batch,num_polar
            loss_mask = self.loss(out_mask, label_primary.argmax(dim=1))  #
            #  uni-encoder
            out_uni = self.primary_uni(uni_out, aspect_pool, len_x)
            loss_uni = self.loss(out_uni, label_primary.argmax(dim=1))

            # loss = F.kl_div(out_aux.log_softmax(dim=-1),
            #                 label_primary.softmax(dim=-1),
            #                 reduction='batchmean')

            loss = loss_full + loss_mask + loss_uni

            return loss, label_primary

    def _freeze_model(self):
        # freeze primary only; encoder is unfreezed
        self.primary.eval()
        for params in self.primary.parameters():
            params.requires_grad = False

    def _unfreeze_model(self):
        self.primary.train()
        for params in self.primary.parameters():
            params.requires_grad = True
