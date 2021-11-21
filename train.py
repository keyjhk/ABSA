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
from config import PARAMETERS, EXCLUDE

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

    def __len__(self):
        return len(self.option)

    def __add__(self, other):
        assert isinstance(other, Option)
        _opt = self.option.copy()
        _opt.update(other.option)
        return Option(_opt, name=self.name + '_' + other.name)

    def __getattr__(self, item):
        try:
            return self.option[item]
        except Exception as e:
            print('key:{} not found'.format(item))
            raise e


class Instructor:
    def __init__(self, opt, logout=True):
        # set seed
        self.reproduce(opt)
        # attr
        self.opt = opt
        self.multi_gpus = True if opt.gpu_parallel and torch.cuda.device_count() > 1 else False
        self.device = 'cuda:{}'.format(opt.device_ids[0]) if self.multi_gpus else opt.device
        # train
        self.batch_size = opt.batch_size if not self.multi_gpus else opt.batch_size * len(opt.device_ids)
        self.max_seq_len = opt.max_seq_len
        self.semi_supervised = opt.semi_supervised
        self.clear_model = opt.clear_model

        # train parameters
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
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.patience = opt.patience
        lr = opt.lr if not self.multi_gpus else opt.lr * len(opt.device_ids)
        self.optimizer = optim.Adam(_params, lr=lr, weight_decay=opt.l2)
        # logger
        self.logout = logout
        self.logger = self.set_logger()

        self.clear()  # clear
        self.load()  # load
        self.tip()

    def tip(self):
        self.logger.info('tips'.center(30, '='))
        self.logger.info('sys:{} multi_gpus:{}-{}'.format(sys.platform, self.multi_gpus, len(self.opt.device_ids)))
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
                              opt=opt
                              ).to(self.device)

        self.model_name = self.model.name
        self.inputs_cols = self.model.inputs_cols
        self._reset_params()
        if self.multi_gpus:
            self.model = torch.nn.DataParallel(self.model, opt.device_ids)

    def _reset_params(self):
        initializers = {
            'xavier_uniform_': torch.nn.init.xavier_uniform_,
            'xavier_normal_': torch.nn.init.xavier_normal_,
            'orthogonal_': torch.nn.init.orthogonal_,
        }
        initailizer = initializers[opt.initializer]
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
                _loss = _loss.mean() if self.multi_gpus else _loss
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

    def save(self, model, epoch, acc, f1):
        save_model_name = self.save_model_name
        states = {
            'model': model.state_dict() if not self.multi_gpus else model.module.state_dict(),
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
            if not self.multi_gpus:
                self.model.load_state_dict(model_cpt['model'])
            else:
                self.model.module.load_state_dict(model_cpt['model'])
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
        acc = round(acc / times, 2)
        f1 = round(f1 / times, 2)
        loss = round(loss / times, 4)

        eval_res = {'loss': loss, 'acc': acc, 'f1': f1}
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
            self.optimizer.zero_grad()
            inputs = [batch[col].to(self.device) for col in self.inputs_cols]
            loss, out, target = self.model(*inputs, mode=mode)
            loss = loss.mean() if self.multi_gpus else loss
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
                acc, f1, loss = self._evaluate_acc_f1(self.validloader)
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


def plot(x, y, xlabel='', ylabel='', title=''):
    # x :[]  y

    plt.xlabel(xlabel)
    plt.xlabel(ylabel)
    title = title if title else '{}-{}-'.format(ylabel, xlabel)
    title += strftime("%m%d-%H%M%S", localtime())
    plt.title(title)
    plt.plot(x, y, marker='o')

    # annotate
    for px, py in zip(x, y):
        plt.annotate(text=str(py), xy=(px, py), xytext=(px, py + 0.1))

    plt.savefig('state/figures/{}.png'.format(title))
    plt.show()


def parameter_explore(opt, par_vals, datasets=['laptop'], semi_sup_compare=False):
    # par_vals:{parameter_name:[val1,],...}
    def is_valid_option(opt):
        is_valid_opt = True
        logger.info('comparing……'.center(30, '='))

        valid_ratios = [0.1, 0.5, 0.7]
        # only these factors influence supervised
        valid_opt_name = '{lr}_{l2}_{drop_lab}'.format(lr=opt.lr, l2=opt.l2, drop_lab=opt.drop_lab)
        if not semi_valid_ratio_scores.get(valid_opt_name):
            semi_valid_ratio_scores[valid_opt_name] = {x: None for x in valid_ratios}
        semi_scores = semi_valid_ratio_scores.get(valid_opt_name)  # {ratio:acc}

        for valid_ratio in valid_ratios:
            # the difference between sup/sems is only the toggle of semi_supervised
            # keep other settings same:drop_lab/drop_unlab/mask_ratio
            # sup
            opt = opt.set({'valid_ratio': valid_ratio})

            if not semi_scores.get(valid_ratio):
                _ins_sup = Instructor(opt.set({'semi_supervised': False}), logout=False)
                res_sup = _ins_sup.run()
                semi_scores[valid_ratio] = res_sup
            else:
                res_sup = semi_scores.get(valid_ratio)

            # semi
            _ins_semi = Instructor(opt.set({'semi_supervised': True}), logout=False)
            res_semi = _ins_semi.run()

            logger.info(
                'valid_ratio: {} semi_acc:{} super_acc :{}'.format(valid_ratio, res_semi['acc'], res_sup['acc'], ))
            if res_semi['acc'] < res_sup['acc']:
                logger.info('option:{} is dropped!'.format(opt.name).center(30, '='))
                is_valid_opt = False
                break
        else:
            logger.info('option:{} is valid!'.format(opt.name).center(30, '='))
        return is_valid_opt

    logger_fname = 'p'
    for p in par_vals.keys():
        logger_fname += '_{}{}'.format(p, len(par_vals[p]))  # p_{parameter_name}{search_len}
    logger = set_logger(name='parameter_explore',
                        file='{}_{}.log'.format(logger_fname, strftime("%m%d-%H%M", localtime())))

    search_options = []
    if len(par_vals) == 0:
        search_options.append(opt)
    elif len(par_vals) == 1:
        # {p:[v1,]}
        for p, values in par_vals.items():
            for v in values:
                tmp_opt = opt.set({
                    p: v,
                }, 'p_' + '{name}[{value}]'.format(name=p, value=v))
                search_options.append(tmp_opt)

    else:
        # random search
        max_len = 50
        hy_params_his = set()  # history
        for i in range(max_len):
            opt_name = 'p_'
            hy_params = {}
            for p, vals in par_vals.items():
                v = random.sample(vals, 1)[0]
                hy_params[p] = v
                opt_name += '{name}[{value}]_'.format(name=p, value=v)
            if opt_name not in hy_params_his:  # check if has created before
                hy_params_his.add(opt_name)
                search_options.append(opt.set(hy_params, name=opt_name))
    if semi_sup_compare:
        semi_valid_ratio_scores = {}  # saved different scores with same options
        logger.warning('have opened the semi_sup_compare!!!')
    search_results = {d: [] for d in datasets}
    best_result = 0
    best_params = None
    for dataset in datasets:
        results = []
        for search_option in search_options:
            _ins = Instructor(search_option.set({'dataset': dataset}))
            res = _ins.run()
            results.append(res)
            vr = 'multi_gpu:{} [dataset]:{dataset} [{option}] [result]:{result}'.format(_ins.multi_gpus,
                                                                                        dataset=dataset,
                                                                                        option=search_option.name,
                                                                                        result=res)
            if semi_sup_compare and not is_valid_option(search_option.set({'dataset': dataset})):
                continue

            search_results[dataset].append(vr)
            logger.info(vr)
            acc = res['acc']
            if acc > best_result:
                best_result = acc
                best_params = vr

    logger.info('final results'.center(30, '*'))
    logger.info('sys:{}'.format(sys.platform, ))
    for d, res in search_results.items():
        # d:dataset res:List[Str]
        for r in sorted(res): logger.info(r)
    logger.info('*' * 30)
    logger.info('best params：{}'.format(best_params))


def main(opt):
    instrutor = Instructor(opt)
    instrutor.run()


if __name__ == '__main__':
    opt = Option(PARAMETERS)
    # supervised
    opt_res = opt.set({'dataset': 'restaurant'}, name='res')  # default supervised
    opt_lap = opt.set({'dataset': 'laptop'}, name='lap')
    # semi_supervised
    opt_semi_res = opt.set({'dataset': 'restaurant', 'semi_supervised': True}, name='semi_res')
    opt_semi_lap = opt.set({'dataset': 'laptop', 'semi_supervised': True}, name='semi_lap')

    # p
    ps = {
        # 'batch_size': [32, 64],
        # 'lr': [2e-3, 1e-3],
        # 'l2': [1e-2, 5e-3, 1e-3],
        # 'threshould': range(4, 20, 2),
        # 'mask_ratio': [x / 10 for x in range(4, 6)],
        # 'drop_lab': [x / 10 for x in range(1, 6)],
        # 'drop_unlab': [x / 10 for x in range(1, 8)],
        # 'drop_attention': [x / 10 for x in range(2, 10, 1)],
        # 'unlabeled_loss': ['mask_weak','mask_strong','all'],
        'valid_ratio': [x / 10 for x in range(0, 10, 2)],
        # 'semi_supervised': [True, False],
        # 'gpu_parallel':[True,False],
        # 'use_weight': [False, True]
    }

    datasets = opt.datasets.keys()
    # parameter_explore(opt, ps)  # super default lap
    # parameter_explore(opt, ps, datasets=datasets)  # super all
    # parameter_explore(opt, ps, datasets=['restaurant'])  # restaurant

    # parameter_explore(opt.set({"semi_supervised": True}), ps,
    #                   semi_sup_compare=True,
    #                   datasets=['laptop'])  # semi default laptop restaurant
    parameter_explore(opt.set({"semi_supervised": True}), ps,datasets=['restaurant'])  # semi default lap
    # parameter_explore(opt.set({"ssemi_supervised": True}), ps,datasets=datasets)  # semi all
