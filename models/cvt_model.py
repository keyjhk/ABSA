import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import *
from models.primary import *
from layers.label_smooth import LabelSmoothing

class CVTModel(nn.Module):
    def __init__(self,
                 num_pos,
                 num_polar,
                 num_position,
                 pretrained_embedding,
                 tokenizer,
                 ### parameters for model ###
                 word_embedding_size,
                 pos_embedding_size,
                 polar_embedding_size,
                 position_embedding_size,
                 encoder_hidden_size,
                 # dynamic features
                 threshould,
                 drop_attention,
                 mask_ratio,
                 weight_alpha,
                 weight_keep,
                 # cvt
                 unlabeled_loss,
                 loss_alpha,
                 mode='labeled',
                 ):

        super().__init__()

        self.inputs_cols = [
            'context_indices', 'pos_indices', 'polar_indices',
            'text_indices', 'position_indices',
            'aspect_indices', 'aspect_boundary',
            'polarity',
            'len_s',
        ]

        self.mode = mode
        self.tokenizer = tokenizer
        # dynamic features
        self.threshould = threshould
        self.mask_ratio = mask_ratio
        self.weight_alpha = weight_alpha
        self.weight_keep = weight_keep
        self.drop_attention = drop_attention
        # cvt
        self.unlabeled_loss = unlabeled_loss
        self.loss_alpha = loss_alpha
        # embedding for word/pos/polar
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.tensor(pretrained_embedding, dtype=torch.float))
        self.pos_embedding = nn.Embedding(num_pos, pos_embedding_size)
        self.polar_embedding = nn.Embedding(num_polar, polar_embedding_size)
        self.position_embedding = nn.Embedding(num_position + 1, position_embedding_size)

        # encoder
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder = BilayerEncoderP(word_embed_dim=word_embedding_size,
                                       position_embed_dim=position_embedding_size,
                                       hidden_size=encoder_hidden_size,
                                       )

        # primary

        self.primary = PolarDecoder(word_embedding=self.word_embedding,
                                    hidden_size=encoder_hidden_size,
                                    num_polar=num_polar,
                                    tokenizer=tokenizer)

        # auxiliary
        self.primary_full = self.primary.clone('full')
        self.mask_weak = self.primary.clone('mask_weak', drop_attention)
        self.mask_strong = self.primary.clone('mask_strong')
        self.primary_weight = self.primary.clone('weight')

        # location decoder
        self.location_decoder = LocationDecoder(self.word_embedding,
                                                encoder_hidden_size,
                                                tokenizer=tokenizer)

        self.name = self.encoder.name

        # loss
        self.loss = LabelSmoothing(0.1)
        # self.loss = nn.CrossEntropyLoss()

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
        polarity = inputs[self.inputs_cols.index('polarity')]  # batch
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
        aspect_pool = self.pool_aspect(aspect_indices, aspect_boundary)  # batch,1,embed_dim

        # uni_out
        uni_out, uni_hidden = self.encoder(word, position, len_x)
        uni_forward = uni_out[:, :, :self.encoder_hidden_size]
        uni_backward = uni_out[:, :, self.encoder_hidden_size:]
        uni_primary = uni_forward + uni_backward
        uni_mask = self.dynamic_features(uni_primary, position_indices, len_x)
        uni_weight = self.dynamic_features(uni_primary, position_indices, len_x, kind='weight')

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

            # docoder primary
            out = self.primary(uni_primary, uni_hidden)
            loss = self.loss(out, polarity)
            return loss, out, polarity
        elif mode == "unlabeled":  # 无监督训练
            self._freeze_model()
            label_primary = self.primary(uni_primary, uni_hidden)  # batch,num_polar

            label_primary = label_primary.detach()
            # mask strong
            out_ms = self.mask_strong(uni_mask, uni_hidden)  # batch,num_polar
            loss_ms = F.kl_div(out_ms.log_softmax(dim=-1), label_primary.softmax(dim=-1), reduction='batchmean')

            # mask weak  drop attention
            out_mw = self.mask_weak(uni_primary, uni_hidden)  # batch,num_polar
            loss_mw = F.kl_div(out_mw.log_softmax(dim=-1), label_primary.softmax(dim=-1), reduction='batchmean')

            # dynamic weight
            out_weight = self.primary_weight(uni_weight, uni_hidden)
            loss_weight = F.kl_div(out_weight.log_softmax(dim=-1), label_primary.softmax(dim=-1), reduction='batchmean')



            if self.unlabeled_loss == 'mask_weak':
                loss = loss_mw
            elif self.unlabeled_loss == 'mask_strong':
                loss = loss_ms
            elif self.unlabeled_loss == 'weight':
                loss = loss_weight
            elif self.unlabeled_loss == 'all':
                loss = loss_weight + loss_mw
                # loss = self.loss_alpha* loss_ms + (1-self.loss_alpha)* loss_mw
                # loss = self.loss_alpha* loss_ms + (1-self.loss_alpha)* loss_weight
            else:
                raise Exception('invalid unlabeled loss')

            return loss, label_primary, polarity

    def dynamic_features(self, features, position_indices, len_x,
                         kind='mask'):
        threshould = self.threshould
        mask_ratio = self.mask_ratio
        weight_alpha = self.weight_alpha
        weight_keep = self.weight_keep
        max_seq_len = self.tokenizer.max_seq_len

        # features: batch,seq_len,hidden_size
        # position: batch,MAX_LEN  ; len_x: batch
        # a vision radium of a sentence
        mb = features.shape[0]
        device = position_indices.device
        max_x = len_x.max()

        # cal window size
        window_size = torch.floor_divide(len_x, 1).unsqueeze(1)  # batch,1
        threshould = torch.tensor([threshould] * mb, device=device).unsqueeze(1)  # batch,1;
        window_size = torch.cat((window_size, threshould), dim=-1)  # batch,2
        window_size = window_size.min(dim=-1, keepdim=True)[0]  # batch,1
        if kind == 'mask':
            # mask over threshoulds
            mask_r = torch.rand(position_indices.shape, device=device)  # batch,MAX_LEN
            mask_r = mask_r.masked_fill(mask_r < mask_ratio, 0)
            mask_r = mask_r.masked_fill(mask_r >= mask_ratio, 1)
            mask_r = mask_r.masked_fill(position_indices <= window_size, 1)
            mask_r = mask_r[:, :max_x].unsqueeze(dim=2)
            return features * mask_r  # batch,seq,hidden_size
        elif kind == 'weight':
            # dynamic weight decay
            # weight = (1 - torch.div(position_indices, max_seq_len)).to(device)  # batch,MAX_LEN ; MAX_LEN = 85
            weight = weight_alpha * (1 - torch.div(position_indices, len_x.unsqueeze(1))).to(device)  # batch,MAX_LEN
            if weight_keep:
                weight = weight.masked_fill(position_indices <= window_size, 1)  # keep 1
            weight = weight[:, :max_x].unsqueeze(dim=2)  # batch,seq_len ,1
            return features * weight
        else:
            raise Exception('error dynamic kind ')

    def pool_aspect(self, aspect_indices, aspect_boundary):
        # aspect embedding average pool
        aspect_len = aspect_boundary[:, 1] - aspect_boundary[:, 0] + 1  # batch
        aspect = self.word_embedding(aspect_indices)  # batch,MAX_LEN,word_embed_dim
        aspect_pool = torch.div(torch.sum(aspect, dim=1),
                                aspect_len.unsqueeze(1)).unsqueeze(1)  # batch,1,embed_dim

        return aspect_pool

    def _freeze_model(self):
        # freeze primary only; encoder is unfreezed
        self.primary.eval()
        self.primary.decoder.train()  # exclude decoder
        for name,params in self.primary.named_parameters():
            params.requires_grad = False

    def _unfreeze_model(self):
        self.primary.train()
        for params in self.primary.parameters():
            params.requires_grad = True
