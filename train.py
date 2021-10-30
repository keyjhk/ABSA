import re
import logging

import torch.nn as nn
import torch.optim as optim
from time import strftime, localtime
from sklearn import metrics
from torch.utils.data import random_split, DataLoader

from data_utils import *
from models.cvt_model import CVTModel
from config import PARAMETERS, EXCLUDE


class Option:
    def __init__(self, options, name='default'):
        self.option = options
        self.name = name
        self._exclude_iter = EXCLUDE

    def set(self, new_options, name=''):
        if not isinstance(new_options, dict):
            raise Exception('options should be a dict')
        _option = self.option.copy()

        for key, val in new_options.items():
            _option[key] = val
        name = name if name else self.name

        return Option(_option, name)

    def __iter__(self):
        keys = self.option.keys()
        for key in keys:
            if key in self._exclude_iter: continue
            yield '[ {} ]:{}'.format(key, self.option[key])

    def __getattr__(self, item):
        try:
            return self.option[item]
        except Exception as e:
            print('key:{} not found'.format(item))
            raise e


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.max_seq_len = opt.max_seq_len
        self.semi_supervised = opt.semi_supervised
        self.clear_model = opt.clear_model

        # train parameters
        self.device = opt.device
        self.start_epoch = 0
        self.max_val_acc = 0
        self.max_val_f1 = 0
        # dataset/dataloader

        self.dataset_files = opt.datasets
        self.datasetname = opt.dataset
        self.valid_ratio = opt.valid_ratio
        self.trainset, self.validset, self.testset, self.unlabeledset = None, None, None, None

        self.trainloader, self.validloader, self.testloader = None, None, None
        self.mixloader = None
        self.tokenizer = None
        self.pretrain_embedding = None
        self.save_model_name = opt.save_model_name
        # model
        self.model = None
        self.inputs_cols = None
        self.best_model = None
        self.model_name = None
        # init
        self.init_dataset()
        self.init_model()
        self.loss = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.semi_supervised:
            # 1e-3过拟合 5e-3 欠拟
            self.patience = opt.patience
            self.optimizer = optim.Adam(_params, lr=opt.lr, weight_decay=opt.l2)
        else:
            # 1e-3过拟合 原1e-2
            self.patience = opt.semi_patience
            self.optimizer = optim.Adam(_params, lr=opt.semi_lr, weight_decay=opt.semi_l2)
        # logger
        self.logger = self.set_logger()

        self.clear()  # clear
        self.load()  # load
        self.tip()

    def tip(self):
        self.logger.info('tips'.center(30, '='))
        opt = self.opt
        for x in opt:
            self.logger.info(x)

        dataset_info = 'init dataset:{} semi:{}' \
                       '\ntrain:{} valid:{} unlabeled:{} test:{}'.format(
            self.datasetname, self.semi_supervised,
            len(self.trainset), len(self.validset), len(self.unlabeledset), len(self.testset)
        )
        self.logger.info(dataset_info)
        self.logger.info('=' * 30)

    def init_dataset(self):
        max_seq_len = self.max_seq_len
        valid_ratio = self.valid_ratio
        batch_size = self.batch_size

        tokenizer = build_tokenizer(max_seq_len=max_seq_len)
        self.tokenizer = tokenizer
        self.pretrain_embedding = build_embedding_matrix(tokenizer.word2idx)

        train_fname = self.dataset_files[self.datasetname]['train']
        test_fname = self.dataset_files[self.datasetname]['test']
        unlabel_fname = self.dataset_files[self.datasetname]['unlabeled']
        trainset = ABSADataset(fname=train_fname, tokenizer=tokenizer)
        testset = ABSADataset(fname=test_fname, tokenizer=tokenizer)

        unlabelset = trainset if not self.semi_supervised else ABSADataset(fname=unlabel_fname, tokenizer=tokenizer)

        if valid_ratio > 0:
            valset_len = int(len(trainset) * valid_ratio)
            trainset, validset = random_split(trainset, [len(trainset) - valset_len, valset_len])
        else:
            validset = testset

        # dataset
        self.trainset = trainset
        self.validset = validset
        self.testset = testset
        self.unlabeledset = unlabelset

        # dataloader
        self.trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        self.validloader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True)
        self.testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
        unlabeled_loader = DataLoader(dataset=unlabelset, batch_size=batch_size, shuffle=True)

        self.mixloader = MixDataLoader(labeled_loader=self.trainloader,
                                       unlabeld_loader=unlabeled_loader, semi_supervised=self.semi_supervised)

    def init_model(self):
        opt = self.opt
        tokenizer = self.tokenizer
        self.model = CVTModel(num_pos=len(tokenizer.pos2idx),
                              num_polar=len(tokenizer.polar2idx),
                              num_position=tokenizer.max_seq_len,
                              pretrained_embedding=self.pretrain_embedding,
                              tokenizer=tokenizer,
                              # encoder
                              word_embedding_size=opt.word_embedding_size,
                              pos_embedding_size=opt.pos_embedding_size,
                              polar_embedding_size=opt.polar_embedding_size,
                              position_embedding_size=opt.position_embedding_size,
                              encoder_hidden_size=opt.encoder_hidden_size,
                              # dynamic mask/weight
                              threshould=opt.threshould,
                              mask_ratio=opt.mask_ratio
                              ).to(self.device)
        self.model_name = self.model.name
        self.inputs_cols = self.model.inputs_cols
        self._reset_params()

    def _reset_params(self):
        print('reset params:'.center(30, '='))
        for name, p in self.model.named_parameters():
            if 'word_embedding' in name:
                print('skip reset parameter: {}'.format(name))
                continue
            if p.requires_grad:
                if len(p.shape) > 1:
                    # torch.nn.init.xavier_uniform_(p)
                    torch.nn.init.xavier_normal_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

        print('=' * 30)

    def set_logger(self):
        logger_file = '{}_{}_{}_{}.log'.format(self.opt.name,
                                               self.datasetname,
                                               self.model_name,
                                               strftime("%m%d-%H%M", localtime()))
        return set_logger(name='Instructor', file=logger_file)

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
        save_model_name = self.save_model_name
        states = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'acc': acc,
            'f1': f1,
        }
        fname = save_model_name.format(dataset=self.datasetname,
                                       model=self.model_name, epoch=epoch,
                                       acc=acc, f1=f1
                                       )

        self.best_model = fname
        torch.save(states, open('state/' + fname, 'wb'))

    def get_model_cpt(self):
        model_pattern = r'{}_{}_epoch.+'.format(self.datasetname, self.model_name)
        return list(filter(lambda x: re.match(model_pattern, x),
                           os.listdir('state')))

    def load(self):
        # '{dataset}_{model}_epoch{epoch}_acc_{acc:.2f}_f1_{f1:.2f}.pkl'
        model_cpt = self.get_model_cpt()
        # sorted by acc
        model_cpt.sort(key=lambda x: float(re.match(r'.+acc_(.+?)_.+', x).group(1)))

        if len(model_cpt) > 0:
            best_model_name = model_cpt[-1]
            self.logger.info('load best_model_name:{}'.format(best_model_name))
            model_cpt = torch.load(open('state/' + best_model_name, 'rb'))
            # max
            self.start_epoch = model_cpt['epoch']
            self.max_val_f1 = model_cpt['f1']
            self.max_val_acc = model_cpt['acc']
            # model/opt
            self.model.load_state_dict(model_cpt['model'])
            self.optimizer.load_state_dict(model_cpt['optimizer'])

    def clear(self):
        self.logger.info('remove saved models'.center(30, '='))
        if not self.clear_model: return
        model_cpt = self.get_model_cpt()
        model_cpt = [os.path.join('state', x) for x in model_cpt]
        if model_cpt:
            self.logger.info('=' * 30)
            for p in model_cpt:
                os.remove(p)
                self.logger.info('remove {}'.format(p))
        self.logger.info('=' * 30)

    def eval(self):
        # in testloader
        acc, f1, loss = 0, 0, 0
        times = self.opt.eval_times
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
        return 'loss:{:.4f} acc:{:.2f}% f1:{:2f}'.format(loss, acc, f1)

    def run(self):
        self.logger.info('run models'.center(30, '='))
        step_every = self.opt.step_every
        print_every = self.opt.print_every

        max_val_acc = self.max_val_acc
        max_val_f1 = self.max_val_f1
        max_val_epoch = self.start_epoch
        epoch = self.start_epoch

        label_steps = self.mixloader.label_len  # how many iters in the labeled dataset
        train_steps = max(1, int(step_every * label_steps))  # how many steps then calculate the train acc/loss
        step = 0
        n_correct, n_total, loss_total = 0, 0, 0
        for batch, mode in self.mixloader.alternating_batch():
            if mode == 'labeled':
                step += 1
            self.model.train()
            self.optimizer.zero_grad()
            inputs = [batch[col].to(self.device) for col in self.inputs_cols]
            loss, out = self.model(*inputs, mode=mode)
            loss.backward()
            self.optimizer.step()

            # progress
            if mode == 'unlabeled': continue  # skip eval
            # loss,acc on train , 1 step 1 show
            percent = 100 * (step % label_steps) / label_steps
            targets = batch['target'].to(self.device)
            n_correct += (torch.argmax(out, -1) == targets).sum().item()
            n_total += len(out)
            loss_total += loss.item()

            if step % label_steps == 0:
                epoch += 1

            if step % train_steps == 0:
                train_acc = 100 * n_correct / n_total
                train_loss = loss_total / train_steps
                n_correct, n_total, loss_total = 0, 0, 0
                print('percent:{:.2f}%, loss:{:.4f}, acc:{:.2f}%'.format(percent, train_loss, train_acc))
                print('-' * 30)

            # evaluate in valid
            if step % (print_every * label_steps) == 0:
                acc, f1, loss = self._evaluate_acc_f1(self.validloader)
                info = 'epoch:{} loss:{:.4f} acc:{:.2f}% f1:{:2f} '.format(epoch, loss, acc, f1)
                print('=' * 30,
                      '\nVALID EVAL\n',
                      info + '\n',
                      '=' * 30)
                self.logger.info(info)  # >> to logger.txt

                if acc > max_val_acc:
                    max_val_acc = acc
                    max_val_epoch = epoch
                    self.save(self.model, max_val_epoch, acc, f1)
                if f1 > max_val_f1:
                    max_val_f1 = f1

            if epoch - max_val_epoch > self.patience:
                self.logger.info('early stop')
                break

        self.load()
        return self.eval()


def set_logger(name=None, file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if file:
        file = 'state/log/' + file
        log_format = '%(asctime)s - %(levelname)s: %(message)s'
        file_hander = logging.FileHandler(file, mode='w', encoding='utf8')
        file_hander.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_hander)  # handler

    logger.addHandler(logging.StreamHandler())  # console
    return logger


def parameter_explore(opt, p, values):
    logger = set_logger(name='parameter explore'
                        , file='parameters_{}_{}.log'.format(p, strftime("%m%d-%H%M", localtime())))
    results = []
    logger.info('parameters:{} explore!'.format(p).center(30, '*'))
    for v in values:
        _opt = opt.set({
            p: v
        }, opt.name + '_{}_{}'.format(p, v))
        _ins = Instructor(_opt)
        res = _ins.run()
        results.append((v, res))

    for v, r in results:
        logger.info('[{}]={} [res]:{}'.format(p, v, r))
    logger.info('*' * 30)


def main():
    # supervised
    opt_su_res = opt.set({
        'semi_supervised': False,
        'dataset': 'restaurant'
    }, name='sup_res')

    opt_su_lap = opt.set({
        'semi_supervised': False,
        'dataset': 'laptop'
    }, name='sup_lap')
    # semi_supervised
    opt_semi_res = opt.set({
        'semi_supervised': True,
        'dataset': 'restaurant'
    }, name='semi_res')

    opt_semi_lap = opt.set({
        'semi_supervised': True,
        'dataset': 'laptop'
    }, name='semi_lap')

    instrutor = Instructor(opt_su_res)
    # instrutor = Instructor(opt_su_lap)
    # instrutor = Instructor(opt_semi_res)
    # instrutor = Instructor(opt_semi_lap)

    instrutor.run()


if __name__ == '__main__':
    opt = Option(PARAMETERS)

    parameter_explore(opt=opt.set({
        'semi_supervised': True,
        'dataset': 'restaurant',
        'valid_ratio': 0.5
    }, name='semi_res'),
        p='threshould',
        values=list(range(4, 6)))
    # main()
