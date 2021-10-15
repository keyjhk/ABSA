import re
import logging
import torch
import torch.optim as opt
from sklearn import metrics
from torch.utils.data import random_split, DataLoader

from data_utils import *
from cvt_model import CVTModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 100
PRINT_EVERY = 1
SAVA_EVERY = 10


class Instructor:
    def __init__(self, batch_size=32, max_seq_len=85, valid_ratio=0.1, mode='alsc', dataset='Laptops'):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = DEVICE
        self.epoches = EPOCH
        self.start_epoch = 1
        self.print_every = PRINT_EVERY
        self.save_every = SAVA_EVERY
        self.mode = mode

        # logger
        self.logger = self.set_logger()

        # initial data
        tokenizer = build_tokenizer(fnames='data/semeval14', max_seq_len=max_seq_len)
        embedding_matrix = build_embedding_matrix(tokenizer.word2idx)
        self.tokenizer = tokenizer

        dataset_fname = 'data/semeval14/{}_Train.xml.seg'.format(dataset)
        self.trainset = ABSADataset(fname=dataset_fname, tokenizer=tokenizer)
        self.testset = ABSADataset(fname=dataset_fname, tokenizer=tokenizer)
        valset_len = int(len(self.trainset) * valid_ratio)
        self.trainset, self.validset = random_split(self.trainset, [len(self.trainset) - valset_len, valset_len])
        self.trainloader = DataLoader(dataset=self.trainset, batch_size=batch_size)
        self.validloader = DataLoader(dataset=self.validset, batch_size=batch_size)
        self.testloader = DataLoader(dataset=self.testset, batch_size=batch_size)

        # initial model
        self.model = CVTModel(num_words=len(tokenizer.word2idx),
                              num_pos=len(tokenizer.pos2idx),
                              num_polar=len(tokenizer.polar2idx),
                              pretrained_embedding=embedding_matrix,
                              tokenizer=self.tokenizer,
                              device=self.device,
                              mode=self.mode
                              ).to(self.device)
        self.inputs_cols = self.model.inputs_cols
        # load model state
        model_cpt = list(filter(lambda x: re.match(r'model_epoch\w+', x), os.listdir('state')))
        model_cpt = model_cpt[-1] if len(model_cpt) > 0 else None
        if model_cpt:
            model_cpt = torch.load(open('state/' + model_cpt, 'rb'))
            model_state = model_cpt['model']
            self.start_epoch = model_cpt['epoch']
            self.model.load_state_dict(model_state)

        # inital optimizer
        self.optimizer = opt.Adam(self.model.parameters(), lr=1e-3)

    @classmethod
    def set_logger(cls):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        Format = '%(asctime)s - %(levelname)s: %(message)s'
        file_hander = logging.FileHandler('state/logger.txt', mode='w', encoding='utf8')
        file_hander.setFormatter(logging.Formatter(Format))

        logger.addHandler(file_hander)
        return logger

    def _evaluate_acc_f1(self):
        dataloader = self.validloader
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        loss = []
        for batch in dataloader:
            # src:[batch*context,batch*pos,batch*polar]
            # target:batch,max_len ;len_x/leny:batch
            # mask_x/mask_y:batch,max_len

            inputs = [batch[col].to(self.device) for col in self.inputs_cols]
            with torch.no_grad():
                loss.append(
                    self.model(*inputs).item()
                )
                # batch,len?
                t_targets, t_outputs = self.model.evaluate_acc_f1(*inputs)
                n_correct += (t_outputs == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1, 2],
                              average='macro')
        loss = sum(loss) / len(loss)
        return acc * 100, f1 * 100, loss

    def save(self, model, epoch, acc, f1):
        states = {
            'model': model.state_dict(),
            'epoch': epoch,
            'acc': acc,
            'f1': f1,
        }
        fname = 'model_epoch{}_acc_{:.2f}_f1_{:.2f}.pkl'.format(epoch, acc, f1)
        torch.save(states, open('state/' + fname, 'wb'))

    def run(self):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0

        for i in range(self.start_epoch, self.epoches + 1):
            print('epoch:{}'.format(i))
            print('=' * 100)
            for p, batch in enumerate(self.trainloader):
                inputs = [batch[col].to(self.device) for col in self.inputs_cols]
                self.optimizer.zero_grad()
                loss = self.model(*inputs)
                loss.backward()

                percent = 100 * p / len(self.trainloader)
                print('percent:{:.2f}% loss:{:.4f}'.format(percent, loss))
                print('-' * 30)
                self.optimizer.step()

            if i % self.print_every == 0:
                acc, f1, loss = self._evaluate_acc_f1()
                print('=' * 30)
                print('EVAL')
                info = 'epoch:{} acc:{:.2f} f1:{:2f} loss:{:.4f}'.format(i, acc, f1, loss)
                print(info)
                self.logger.info(info)  # >> to logger.txt
                print('=' * 30)

                if acc > max_val_acc:
                    max_val_acc = acc
                    max_val_epoch = i
                    self.save(self.model, max_val_epoch, acc, f1)
                if f1 > max_val_f1:
                    max_val_f1 = f1


if __name__ == '__main__':
    instrutor = Instructor(mode='alsc', dataset='Restaurants')
    instrutor.run()
