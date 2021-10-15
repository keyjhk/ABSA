import torch
import torch.nn as nn
from encoder import Encoder
from primary import Decoder


class CVTModel(nn.Module):
    def __init__(self,
                 num_words,
                 num_pos,
                 num_polar,
                 pretrained_embedding,
                 tokenizer,
                 device,
                 mode,
                 ### parameters for encoder goes here ###
                 word_embedding_size=300,
                 pos_embedding_size=50,
                 dropout_lab=0.5,
                 encoder_hidden_size=300,

                 ):

        super().__init__()

        # inputs cols:
        self.inputs_cols = [
            'context_indices', 'pos_indices', 'polar_indices',
            'target',
            'len_s', 'len_t',
            'mask_s', 'mask_t'
        ]

        # tokenizer
        self.tokenizer = tokenizer
        self.device = device
        self.mode = mode
        # embedding
        self.word_embedding = nn.Embedding(num_words, word_embedding_size)
        self.word_embedding.from_pretrained(torch.tensor(pretrained_embedding))
        self.polar_embedding = nn.Embedding(num_pos, encoder_hidden_size)
        # encoders parameters
        self.encoder = Encoder(num_words=num_words,
                               num_pos=num_pos,
                               num_polar=num_polar,
                               word_embedding=self.word_embedding,
                               polar_embedding=self.polar_embedding,
                               pos_embedding_size=pos_embedding_size,
                               dropout_lab=dropout_lab,
                               hidden_size=encoder_hidden_size,
                               device=self.device
                               )

        # decoder  parameters
        self.primary = Decoder(num_polar=num_polar,
                               word_embedding=self.word_embedding,
                               polar_embedding=self.polar_embedding,
                               hidden_size=encoder_hidden_size,
                               tokenizer=tokenizer,
                               device=self.device,
                               mode=mode)

    def evaluate_acc_f1(self, context, pos, polar, target, len_x, len_y, mask_x, mask_y):
        # target:batch,max_len ; mask:batch,max_len  len_x: batch
        max_t = len_y.max()
        polar_flag = self.primary.build_polar_flag(max_t)
        predict = self.forward(context, pos, polar, target, len_x, len_y, mask_x, mask_y,
                               predict=True).argmax(dim=-1).transpose(0, 1)  # batch,max_t

        # need to convert polar_index to 0/1/2
        predict = predict % len_x.unsqueeze(1)
        _target, _predict = None, None
        # 两个任务在返回的时间步大小上不一样
        if self.mode == 'aesc':
            _target = target[:, :max_t]  # batch,max_t
            _predict = predict
            _mask_y = mask_y[:, max_t]
            return target[:, :max_t].mask, predict  # batch,max_t
        elif self.mode == 'alsc':
            indices = torch.where(polar_flag == 1)[0]  # select polar indexs
            _target = target[:, indices]
            _predict = predict[:, indices]
            _mask_y = mask_y[:, indices]  # batch,len?

        return torch.masked_select(_target, _mask_y), torch.masked_select(_predict, _mask_y)

    def forward(self, context, pos, polar, target, len_x, len_y, mask_x, mask_y,
                mode='labeled', predict=False):
        # src:[batch*context,batch*pos,batch*polar]
        # target:batch,max_len ;len_x/leny:batch
        # mask_x/mask_y:batch,max_len

        # encoder_out:batch,seq_len,hidden_size*2 ; last_hidden:layers*2,batch,hidden_size
        encoder_out, last_hidden = self.encoder(context, pos, polar, len_x, mask_x)

        if mode == "labeled":  # 监督训练
            self._unfreeze_model()
            if not predict:
                loss = self.primary.calculate_loss(x=context, len_x=len_x,
                                                   encoder_out=encoder_out,
                                                   hidden=last_hidden,
                                                   target=target, len_t=len_y, mask_t=mask_y)

                return loss
            else:
                predict = self.primary.predict(x=context, len_x=len_x,
                                               encoder_out=encoder_out,
                                               hidden=last_hidden,
                                               target=target, len_t=len_y)
                return predict  # max_t,batch,seq_len+num_polar
        elif mode == "unlabeled":  # 无监督训练
            pass
            # self._freeze_model()
            # # calculate the prediction and loss of the  auxiliary modules
            # # loss += loss_full + loss_forwards + loss_backwards + loss_future + loss_past
            # torch.cuda.empty_cache()  # 为什么要empty
            #
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
