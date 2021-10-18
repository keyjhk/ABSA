import re
import logging
import torch
import torch.nn as nn
import torch.optim as opt
from time import strftime, localtime
from sklearn import metrics
from torch.utils.data import random_split, DataLoader

from data_utils import *
from models.cvt_model import CVTModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 50
PRINT_EVERY = 1


class Instructor:
    def __init__(self, batch_size=16, max_seq_len=85, valid_ratio=0,
                 mode='alsc', dataset='laptop', model='atae'):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        # train parameters
        self.device = DEVICE
        self.epoches = EPOCH
        self.start_epoch = 1
        self.step_every = 1
        self.print_every = PRINT_EVERY
        self.mode = mode
        # dataset/dataloader
        self.dataset_files = {
            'restaurant': {
                'train': 'data/semeval14/Restaurants_Train.xml.seg',
                'test': 'data/semeval14/Restaurants_Test_Gold.xml.seg'
            },
            'laptop': {
                'train': 'data/semeval14/Laptops_Train.xml.seg',
                'test': 'data/semeval14/Laptops_Test_Gold.xml.seg'
            }
        }
        self.datasetname = dataset
        self.valid_ratio = valid_ratio
        self.trainset, self.validset, self.testset = None, None, None
        self.trainloader, self.validloader, self.testloader = None, None, None
        self.tokenizer = None
        self.pretrain_embedding = None
        # model
        self.model = None
        self.model_name = model
        self.inputs_cols = None
        self.best_model = None
        # init
        self.init_dataset()
        self.init_model()
        self.loss = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = opt.Adam(_params, lr=1e-3, weight_decay=0.01)  # weight_decay=0.01
        # logger
        self.logger = self.set_logger()

        # load
        self.load()

    def init_dataset(self):
        max_seq_len = self.max_seq_len
        valid_ratio = self.valid_ratio
        batch_size = self.batch_size

        tokenizer = build_tokenizer(fnames='data/semeval14', max_seq_len=max_seq_len)
        self.tokenizer = tokenizer
        self.pretrain_embedding = build_embedding_matrix(tokenizer.word2idx)

        train_fname = self.dataset_files[self.datasetname]['train']
        test_fname = self.dataset_files[self.datasetname]['test']
        trainset = ABSADataset(fname=train_fname, tokenizer=tokenizer)
        testset = ABSADataset(fname=test_fname, tokenizer=tokenizer)
        if valid_ratio > 0:
            valset_len = int(len(trainset) * valid_ratio)
            trainset, validset = random_split(trainset, [len(trainset) - valset_len, valset_len])
        else:
            validset = testset

        self.trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        self.validloader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True)
        self.testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    def init_model(self):
        tokenizer = self.tokenizer
        self.model = CVTModel(num_pos=len(tokenizer.pos2idx),
                              num_polar=len(tokenizer.polar2idx),
                              num_position = tokenizer.max_seq_len,
                              pretrained_embedding=self.pretrain_embedding,
                              combine=self.model_name,
                              ).to(self.device)
        self.inputs_cols = self.model.inputs_cols
        self._reset_params()

    def _reset_params(self):
        for name, p in self.model.named_parameters():
            if 'embed' in name:
                print('skip parameter: {}'.format(name))
                continue
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

            # print('=' * 30)

    def set_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        Format = '%(asctime)s - %(levelname)s: %(message)s'
        logger_file = 'logger_{}_{}_{}.log'.format(self.datasetname, self.model_name,
                                                   strftime("%m%d-%H%M", localtime()))
        logger_file = 'state/' + logger_file
        file_hander = logging.FileHandler(logger_file, mode='w', encoding='utf8')
        file_hander.setFormatter(logging.Formatter(Format))

        logger.addHandler(file_hander)  # handler
        logger.addHandler(logging.StreamHandler())  # console
        return logger

    def _evaluate_acc_f1(self, dataloader):
        self.model.eval()
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        loss = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = [batch[col].to(self.device) for col in self.inputs_cols]
                _loss, out = self.model(*inputs)  # out:batch,num_polar
                loss.append(_loss.item())
                t_targets, t_outputs = batch['target'].to(self.device), out.argmax(dim=-1)  # batch
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
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'acc': acc,
            'f1': f1,
        }
        fname = '{}_epoch{}_acc_{:.2f}_f1_{:.2f}_{}.pkl'.format(self.model_name,
                                                                epoch, acc, f1,
                                                                self.datasetname)
        self.best_model = fname
        torch.save(states, open('state/' + fname, 'wb'))

    def load(self):
        model_pattern = r'{}_epoch\w+'.format(self.model_name)
        model_cpt = list(filter(lambda x: re.match(model_pattern, x),
                                os.listdir('state')))
        model_cpt.sort(key=lambda x: float(x.split('_')[3]))  # sorted by acc

        if len(model_cpt) > 0:
            best_model_name = model_cpt[-1]
            self.logger.info('best_model_name:{}'.format(best_model_name))
            model_cpt = torch.load(open('state/' + best_model_name, 'rb'))
            self.start_epoch = model_cpt['epoch'] + 1
            self.model.load_state_dict(model_cpt['model'])
            self.optimizer.load_state_dict(model_cpt['optimizer'])

    def eval(self, times=5):
        # in testloader
        acc, f1, loss = 0, 0, 0
        for t in range(times):
            _acc, _f1, _loss = self._evaluate_acc_f1(self.testloader)
            acc += _acc
            f1 += _f1
            loss += _loss
        acc /= times
        f1 /= times
        loss /= times
        self.logger.info('TEST EVAL'.center(30, '='))
        self.logger.info('loss:{:.4f} acc:{:.2f}% f1:{:2f}'.format(loss, acc, f1))

    def run(self):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = self.start_epoch

        for i in range(self.start_epoch, self.epoches + 1):
            print('epoch:{}'.format(i))
            print('=' * 100)

            self.model.train()
            n_correct, n_total, loss_total = 0, 0, 0
            for p, batch in enumerate(self.trainloader, start=1):
                self.optimizer.zero_grad()
                inputs = [batch[col].to(self.device) for col in self.inputs_cols]
                loss, out = self.model(*inputs)
                loss.backward()
                self.optimizer.step()

                # loss,train_acc
                percent = 100 * p / len(self.trainloader)
                targets = batch['target'].to(self.device)
                n_correct += (torch.argmax(out, -1) == targets).sum().item()
                n_total += len(out)
                loss_total += loss.item()
                if p % self.step_every == 0:
                    train_acc = 100 * n_correct / n_total
                    train_loss = loss_total / p
                    print('percent:{:.2f}%, loss:{:.4f}, acc:{:.2f}%'.format(percent, train_loss, train_acc))
                    print('-' * 30)

            if i % self.print_every == 0:
                acc, f1, loss = self._evaluate_acc_f1(self.validloader)
                info = 'epoch:{} loss:{:.4f} acc:{:.2f}% f1:{:2f} '.format(i, loss, acc, f1)
                print('=' * 30,
                      '\nVALID EVAL\n',
                      info + '\n',
                      '=' * 30)
                self.logger.info(info)  # >> to logger.txt

                if acc > max_val_acc:
                    max_val_acc = acc
                    max_val_epoch = i
                    self.save(self.model, max_val_epoch, acc, f1)
                if f1 > max_val_f1:
                    max_val_f1 = f1

            if i - max_val_epoch > 10:
                self.logger.info('early stop')
                print('early stop')
                break

        self.load()
        self.eval()


def main():
    input_colses = {
        'atae_lstm': ['text_indices', 'aspect_indices', 'aspect_boundary'],
        'cvt': [
            'context_indices', 'pos_indices', 'polar_indices',
            'aspect_indices', 'aspect_boundary',
            'target', 'len_s',
            'text_indices'
        ]
    }


if __name__ == '__main__':
    instrutor = Instructor(dataset='restaurant',model='cvt-at-position')
    instrutor.run()
    # instrutor.eval()
