import sys
from time import strftime, localtime, time
import logging

import numpy
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import random_split, DataLoader

import matplotlib.pyplot as plt

from data_utils import *
from models.cvt_model import CVTModel
from config import DEFAULT_OPTION

import warnings

warnings.filterwarnings('ignore')


def time_cal(func):
    def inner(*args, **kwargs):
        st = time()
        res = func(*args, **kwargs)
        time_cost = int(time() - st)
        if isinstance(res, dict):
            res['time_cost'] = '{}m{}s'.format(time_cost // 60, time_cost % 60)
        return res

    return inner


class Instructor:
    def __init__(self, opt, logout=True):
        # set seed
        self.reproduce(opt)
        # attr
        self.opt = opt
        self.device = 'cuda:{}'.format(
            opt.device_ids[0]) if 'cuda' in opt.device and torch.cuda.device_count() > 1 else opt.device
        self.run_time = int(time())
        # train
        self.batch_size = opt.batch_size
        self.max_seq_len = opt.max_seq_len
        self.semi_supervised = opt.semi_supervised

        # train parameters
        self.start_epoch = 0
        self.max_val_acc = 0
        self.max_val_f1 = 0
        # dataset/dataloader

        self.dataset_files = opt.datasets
        self.datasetname = opt.dataset
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
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.patience = opt.patience
        self.optimizer = optim.Adam(_params, lr=opt.lr, weight_decay=opt.l2)
        # logger
        self.logout = logout
        self.logger = self.set_logger()

        # self.clear()
        self.load()
        self.tip()

    def tip(self):
        self.logger.info('tips'.center(30, '='))
        self.logger.info('sys:{}'.format(sys.platform, len(self.opt.device_ids)))
        opt = self.opt
        for x in opt:
            self.logger.info(x)
        dataset_info = 'init dataset:{} semi:{}' \
                       '\ntrain:{} valid:{} unlabeled:{} test:{}'.format(
            self.datasetname, self.semi_supervised,
            len(self.trainset), len(self.validset), len(self.unlabeledset), len(self.testset)
        )
        self.logger.info(dataset_info)
        self.logger.info(self.optimizer)
        self.logger.info('=' * 30)

    def reproduce(self, opt):
        if opt.seed is not None:
            random.seed(opt.seed)
            numpy.random.seed(opt.seed)
            torch.manual_seed(opt.seed)
            torch.cuda.manual_seed(opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(opt.seed)

    def init_dataset(self):
        max_seq_len = self.max_seq_len
        valid_ratio = self.opt.valid_ratio
        batch_size = self.batch_size

        tokenizer = build_tokenizer(max_seq_len=max_seq_len)
        self.tokenizer = tokenizer
        self.pretrain_embedding = build_embedding_matrix(tokenizer.word2idx)

        train_fname = self.dataset_files[self.datasetname]['train']
        test_fname = self.dataset_files[self.datasetname]['test']
        unlabel_fname = self.dataset_files[self.datasetname]['unlabeled']
        trainset = ABSADataset(fname=train_fname, tokenizer=tokenizer)
        testset = ABSADataset(fname=test_fname, tokenizer=tokenizer)
        unlabelset = ABSADataset(fname=unlabel_fname, tokenizer=tokenizer)

        # split dataset
        # unlabel
        if self.opt.unlabel_len is not None:
            unlabel_len = int(min(self.opt.unlabel_len, len(unlabelset)))
            _, unlabelset = random_split(unlabelset, [len(unlabelset) - unlabel_len, unlabel_len])

        # train
        if self.opt.train_len is not None:
            train_len = int(min(self.opt.train_len, len(trainset)))
            _, trainset = random_split(trainset, [len(trainset) - train_len, train_len])
        # valid
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
                                       unlabeld_loader=unlabeled_loader,
                                       semi_supervised=self.semi_supervised)

    def init_model(self):
        opt = self.opt
        tokenizer = self.tokenizer

        self.model = CVTModel(num_pos=len(tokenizer.pos2idx),
                              num_polar=len(tokenizer.polar2idx),
                              num_position=tokenizer.max_seq_len,
                              pretrained_embedding=self.pretrain_embedding,
                              tokenizer=tokenizer,
                              opt=opt
                              ).to(self.device)

        self.model_name = self.model.name
        self.inputs_cols = self.model.inputs_cols
        self._reset_params()

    def _reset_params(self):
        initializers = {
            'xavier_uniform_': torch.nn.init.xavier_uniform_,
            'xavier_normal_': torch.nn.init.xavier_normal_,
            'orthogonal_': torch.nn.init.orthogonal_,
        }
        initailizer = initializers[self.opt.initializer]
        # print('reset params:'.center(30, '='))
        for name, p in self.model.named_parameters():
            if 'word_embedding' in name:
                # print('skip reset parameter: {}'.format(name))
                continue
            if p.requires_grad:
                if len(p.shape) > 1:
                    initailizer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

        # print('=' * 30)

    def set_logger(self):
        logger_file = '{}_{}_{}_{}.log'.format(self.opt.name,
                                               self.datasetname,
                                               self.model_name,
                                               strftime("%m%d-%H%M", localtime()))
        logger_name = '{}_{}'.format(self.opt.name, self.model_name)
        level = logging.INFO if self.logout else logging.WARNING
        return set_logger(name=logger_name, file=logger_file, level=level)

    def _evaluate_acc_f1(self, dataloader):
        self.model.eval()
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        loss = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = [batch[col].to(self.device) for col in self.inputs_cols]
                _loss, out, target = self.model(*inputs)  # out:batch,num_polar
                loss.append(_loss.item())
                t_targets, t_outputs = target, out.argmax(dim=-1)  # batch
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

        acc = round(acc * 100, 2)
        f1 = round(f1 * 100, 2)
        loss = round(loss, 4)
        return acc, f1, loss

    def predict(self, name, sample=None):
        # sample : context,aspect,polarity
        saved_dir = 'state/predict/'
        saved_name = '{}_predict_results'.format(''.join(name.split('.')[:-1]))  # remove suffix

        from data_utils import build_indices
        self.load(name)
        model = self.model
        model.eval()
        tokenizer = self.tokenizer
        predict_results = {'true': [], 'false': []}  # (context,aspect,target,prediction)
        n_correct, n_total = 0, 0

        with torch.no_grad():
            if not sample:
                if not os.path.exists(saved_dir + saved_name + '.pkl'):
                    # create predict results and stored then :
                    self.logger.info('create predict results??????')
                    dataloader = self.testloader
                    for batch in dataloader:
                        inputs = [batch[col].to(self.device) for col in self.inputs_cols]
                        _loss, out, target = model(*inputs)  # out:batch,num_polar ; target: batch
                        t_targets, t_outputs = target, out.argmax(dim=-1)  # batch
                        n_correct += (t_targets == t_outputs).sum()
                        n_total += t_targets.shape[0]

                        aspects = [tokenizer.sequence_to_text(indice) for indice in
                                   batch['aspect_indices'].numpy()]
                        batch_contexts = [tokenizer.sequence_to_text(indice) for indice in
                                          batch['context_indices'].numpy()]  # batch
                        # statistic true prediction and false prediction each batch
                        for i in range(target.shape[0]):
                            _c, _a, _t, _o = batch_contexts[i], aspects[i], t_targets[i].item(), t_outputs[i].item()
                            key = 'true' if t_targets[i] == t_outputs[i] else 'false'
                            predict_results[key].append((_c, _a, _t, _o))

                    print('total:{} acc:{} predict_true:{} predict_false:{}'.format(n_total, n_correct * 100 / n_total,
                                                                                    len(predict_results['true']),
                                                                                    len(predict_results['false'])))
                    # export predict results
                    self.logger.info('writing {}.pkl ....'.format(saved_name))
                    self.logger.info('writing {}.txt ....'.format(saved_name))
                    pickle.dump(predict_results, open(saved_dir + saved_name + '.pkl', 'wb'))
                    with open(saved_dir + saved_name + '.txt', 'w') as f:
                        for label, results in predict_results.items():
                            f.write(label.center(30, '=') + '\n')
                            for res in results:
                                text = '\n'.join([str(x) for x in res])  # context ,target,predict
                                text += '\n' * 2
                                f.write(text)
                            f.write('\n' * 3)

                return saved_dir + saved_name + '.pkl'
            else:
                # predict for a given sample
                # format the sentence
                indices = build_indices(tokenizer, *sample, partition_token=sample[1])
                dataloader = DataLoader(dataset=[indices], batch_size=1)
                # for key in indices.keys():  # to tensor
                #     val = indices[key]
                #     if isinstance(val, numpy.ndarray):
                #         if val.size > 1:
                #             indices[key] = torch.tensor(val).view(1, -1)
                #         else:
                #             indices[key] = torch.tensor(val).view(1)
                #     else:
                #         try:
                #             indices[key] = torch.tensor(int(val)).view(1)
                #         except Exception:
                #             pass
                # feed it into the model
                for batch in dataloader:
                    inputs = [batch[col].to(self.device) for col in self.inputs_cols]
                    _loss, out, target = model(*inputs)  # out:batch,num_polar ; target: batch
                    out = out.argmax(dim=-1)
                    print('sentence: {}\naspect: {}\n'
                          'predict:{} target:{}\n'.format(sample[0], sample[1],
                                                        out.item(), target.item()))
                return out.item

    def save(self, model, epoch, acc, f1):
        save_model_name = self.save_model_name
        states = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'acc': acc,
            'f1': f1,
        }
        fname = save_model_name.format(time=self.run_time, dataset=self.datasetname,
                                       model=self.model_name, epoch=epoch,
                                       acc=acc, f1=f1
                                       )

        self.best_model = fname
        torch.save(states, open('state/' + fname, 'wb'))

    def get_model_cpt(self):
        # {dataset}_{time}_{model}_epoch{epoch}_acc_{acc:.2f}_f1_{f1:.2f}.pkl
        # model_pattern = r'{}_{}_epoch.+'.format(self.datasetname, self.model_name)
        model_pattern = r'{}_{}_{}_epoch.+'.format(self.datasetname, self.run_time, self.model_name)
        return list(filter(lambda x: re.match(model_pattern, x),
                           os.listdir('state')))

    def load(self, name=None):
        if name is None:
            # '{dataset}_{model}_epoch{epoch}_acc_{acc:.2f}_f1_{f1:.2f}.pkl'
            model_cpt = self.get_model_cpt()
            # sorted by acc
            model_cpt.sort(key=lambda x: float(re.match(r'.+acc_(.+?)_.+', x).group(1)))
            if len(model_cpt) > 0:
                best_model_name = model_cpt[-1] if len(model_cpt) > 0 else model_cpt
            else:
                return
        else:
            best_model_name = name

        self.logger.info('load best_model_name:{}'.format(best_model_name))
        model_cpt = torch.load(open('state/' + best_model_name, 'rb'))
        # max
        self.start_epoch = model_cpt['epoch']
        self.max_val_f1 = model_cpt['f1']
        self.max_val_acc = model_cpt['acc']
        # model/opt
        self.model.load_state_dict(model_cpt['model'])

        self.optimizer.load_state_dict(model_cpt['optimizer'])
        return best_model_name

    def clear(self):
        self.logger.info('remove saved models'.center(30, '='))
        if not self.opt.clear_model: return
        model_cpt = self.get_model_cpt()
        model_cpt = [os.path.join('state', x) for x in model_cpt]
        if model_cpt:
            self.logger.info('=' * 30)
            for p in model_cpt:
                os.remove(p)
                self.logger.info('remove {}'.format(p))
        self.logger.info('=' * 30)

    def eval(self, dataloader=None):
        dataloader = self.testloader if dataloader is None else dataloader
        acc, f1, loss = 0, 0, 0
        times = self.opt.eval_times
        for t in range(times):
            _acc, _f1, _loss = self._evaluate_acc_f1(dataloader)
            acc += _acc
            f1 += _f1
            loss += _loss
        acc = round(acc / times, 2)
        f1 = round(f1 / times, 2)
        loss = round(loss / times, 4)

        eval_res = {'loss': loss, 'acc': acc, 'f1': f1}
        if dataloader is self.testloader:
            self.logger.info('TEST EVAL'.center(30, '='))
            self.logger.info(str(eval_res))

        return eval_res

    @time_cal
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
        epoch_time_costs = time()
        for batch, mode in self.mixloader.alternating_batch():
            if mode == 'labeled':
                step += 1
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            inputs = [batch[col].to(self.device) for col in self.inputs_cols]
            loss, out, target = self.model(*inputs, mode=mode)
            loss.backward()
            self.optimizer.step()

            # progress
            if mode == 'unlabeled': continue  # skip eval
            # loss,acc on train , 1 step 1 show
            percent = 100 * (step % label_steps) / label_steps
            n_correct += (torch.argmax(out, -1) == target).sum().item()
            n_total += len(out)
            loss_total += loss.item()

            if step % label_steps == 0:
                epoch += 1

            if step % train_steps == 0:
                train_acc = 100 * n_correct / n_total
                train_loss = loss_total / train_steps
                n_correct, n_total, loss_total = 0, 0, 0
                self.logger.info('percent:{:.2f}%, loss:{:.4f}, acc:{:.2f}%'.format(percent, train_loss, train_acc))
                self.logger.info('-' * 30)

            # evaluate in valid
            if step % (print_every * label_steps) == 0:
                valid_res = self.eval(self.validloader)
                acc, f1, loss = valid_res['acc'], valid_res['f1'], valid_res['loss']
                info = 'epoch:{} epoch_time:{}s loss:{} acc:{}% f1:{} '.format(epoch,
                                                                               (time() - epoch_time_costs) // epoch,
                                                                               loss, acc, f1)
                self.logger.info('=' * 30 + '\nVALID EVAL\n' + info + '\n' + '=' * 30)

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
        self.clear()  # clear saved models produced at this running time
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


def main():
    opt = DEFAULT_OPTION
    lap_opt = opt.set({'dataset': 'laptop', 'drop_lab': 0.2, 'window_weight': 2})
    res_opt = opt.set({'dataset': 'restaurant', 'drop_lab': 0.4, 'window_weight': 0})
    instrutor = Instructor(res_opt)
    # instrutor = Instructor(lap_opt)
    instrutor.run()


if __name__ == '__main__':
    main()
