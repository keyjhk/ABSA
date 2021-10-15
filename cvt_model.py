import torch
import torch.nn as nn
from encoder import Encoder
from primary import Primay


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
                 polar_embedding_size=300,
                 encoder_hidden_size=256,

                 ):

        super().__init__()

        # inputs cols: 和 forward所传参数顺序一致
        self.inputs_cols = [
            'context_indices', 'pos_indices', 'polar_indices',
            'aspect_indices','aspect_boundary',
            'target','len_s'
        ]

        # tokenizer
        self.tokenizer = tokenizer
        self.device = device
        self.mode = mode
        # embedding for word/pos/polar
        self.word_embedding = nn.Embedding(num_words, word_embedding_size)
        self.word_embedding.from_pretrained(torch.tensor(pretrained_embedding))
        self.pos_embedding = nn.Embedding(num_pos, pos_embedding_size)
        self.polar_embedding = nn.Embedding(num_pos, polar_embedding_size)
        # encoders parameters
        self.encoder = Encoder(num_words=num_words,
                               num_pos=num_pos,
                               num_polar=num_polar,
                               word_embedding=self.word_embedding,
                               pos_embedding=self.polar_embedding,
                               hidden_size=encoder_hidden_size,
                               )

        # primary  parameters
        self.primary = Primay(hidden_dim=encoder_hidden_size * 2,
                              word_embed_dim=word_embedding_size,
                              polar_dim=polar_embedding_size,
                              num_polar=num_polar)
        # loss
        self.loss = nn.CrossEntropyLoss()


    def forward(self, context, pos, polar,
                aspect_indices, aspect_boundary,
                target, len_x,
                mode='labeled'):
        # context/pos/polar/aspect:batch,MAX_LEN
        # aspect_boundary: batch,2
        # target:batch ;len_x:batch
        # mask_x:batch,MAX_LEN

        # pool(average) aspect
        aspect_len = aspect_boundary[:,1] - aspect_boundary[:,0] + 1  # batch
        aspect = self.word_embedding(aspect_indices)  # batch,MAX_LEN,word_embed_dim
        aspect_pool = torch.div(torch.sum(aspect, dim=1),
                                aspect_len.unsqueeze(1)).unsqueeze(1)  # batch,1,embed_dim
        # polar_embedding
        polar = self.polar_embedding(polar)  # batch,MAX_LEN,word_embed_dim

        # encoder_out:batch,seq_len,hidden_size*2
        encoder_out = self.encoder(context, pos, len_x)

        if mode == "labeled":  # 监督训练
            self._unfreeze_model()
            out = self.primary(encoder_out, polar, aspect_pool, len_x)
            loss = self.loss(out,target)
            return loss,out
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
