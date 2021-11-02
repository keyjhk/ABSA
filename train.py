from time import strftime, localtime
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
    def __init__(self, opt):
        # set seed
        self.reproduce(opt)
        # attr
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
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.semi_supervised:
            self.patience = opt.semi_patience
            self.optimizer = optim.Adam(_params, lr=opt.semi_lr, weight_decay=opt.semi_l2)
        else:
            self.patience = opt.patience
            self.optimizer = optim.Adam(_params, lr=opt.lr, weight_decay=opt.l2)
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
                              # encoder
                              word_embedding_size=opt.word_embedding_size,
                              pos_embedding_size=opt.pos_embedding_size,
                              polar_embedding_size=opt.polar_embedding_size,
                              position_embedding_size=opt.position_embedding_size,
                              encoder_hidden_size=opt.encoder_hidden_size,
                              # dynamic mask/weight
                              threshould=opt.threshould,
                              drop_attention=opt.drop_attention,
                              mask_ratio=opt.mask_ratio,
                              weight_keep=opt.weight_keep,
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
        initailizer = initializers[opt.initializer]
        print('reset params:'.center(30, '='))
        for name, p in self.model.named_parameters():
            if 'word_embedding' in name:
                print('skip reset parameter: {}'.format(name))
                continue
            if p.requires_grad:
                if len(p.shape) > 1:
                    initailizer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

        print('=' * 30)

    def set_logger(self):
        logger_file = '{}_{}_{}_{}.log'.format(self.opt.name,
                                               self.datasetname,
                                               self.model_name,
                                               strftime("%m%d-%H%M", localtime()))
        logger_name = '{}_{}'.format(self.opt.name, self.model_name)
        return set_logger(name=logger_name, file=logger_file)

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
        acc = round(acc / times, 2)
        f1 = round(f1 / times, 2)
        loss = round(loss / times, 4)

        eval_res = {'loss': loss, 'acc': acc, 'f1': f1}
        self.logger.info('TEST EVAL'.center(30, '='))
        self.logger.info(str(eval_res))
        return eval_res

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
                print('percent:{:.2f}%, loss:{:.4f}, acc:{:.2f}%'.format(percent, train_loss, train_acc))
                print('-' * 30)

            # evaluate in valid
            if step % (print_every * label_steps) == 0:
                acc, f1, loss = self._evaluate_acc_f1(self.validloader)
                info = 'epoch:{} loss:{} acc:{}% f1:{} '.format(epoch, loss, acc, f1)
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


def parameter_explore(opt, par_vals, isplot=True, datasets=None):
    # par_vals:{parameter_name:[val1,],...}
    logger_fname = 'p'
    for p in par_vals.keys():
        logger_fname += '_{}{}'.format(p[:3], len(par_vals[p]))  # p_{parameter_name}{search_len}
    logger = set_logger(name='parameter_explore',
                        file='{}_{}.log'.format(logger_fname, strftime("%m%d-%H%M", localtime())))

    datasets = opt.datasets.keys() if datasets is None else datasets
    search_results = {d: [] for d in datasets}
    for dataset in datasets:
        for p, values in par_vals.items():
            pv = 'p_{}{}-{}_{}'.format(p, values[0], values[-1], dataset)  # logger_fname for plot fig
            logger.info(pv.center(30, '*'))

            results = []
            for i, v in enumerate(values):
                # p_opt.name_par.logger_fname[p.index]_p.val_dataset
                _opt = opt.set({
                    p: v,
                    'dataset': dataset,
                }, 'p_' + opt.name + '_{}[{}]_{}_{}'.format(p, i, v, dataset))
                _ins = Instructor(_opt)
                res = _ins.run()
                # add results
                results.append(res)
                vr = '[dataset]:{} [{}]:{} [res]:{}'.format(dataset, p, v, res)
                search_results[dataset].append(vr)
                # show config and result each run
                logger.info('=' * 30)
                for x in _opt: logger.info(x)
                logger.info(vr)
                logger.info('=' * 30)

            # simple show ,no config
            for i in range(len(results), 0, -1): logger.info(search_results[dataset][-i])
            if isplot:
                try:
                    # plot
                    x = [float(v) for v in values]
                    y = [res['acc'] for _, res in results]
                    plot(x=x, y=y, xlabel=p, ylabel='acc', title=pv + '+' + opt.name)
                except Exception:
                    pass

    logger.info('final results'.center(30, '*'))
    for d, res in search_results.items():
        # d:dataset res:List
        for r in res: logger.info(r)
    logger.info('*' * 30)


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
        # 'weight_keep': [False],
        # 'batch_size': [64],
        # 'semi_lr': [5e-3,5e-4,1e-4,1e-5],
        # 'semi_l2': [1e-1, 1e-2, 1e-3],
        # 'l2': [1e-2, 5e-3],
        # 'patience':range(10,40,5),
        # 'pos_embedding_size':range(50,350,50),  # 50
        'threshould': list(range(20,30,2)),
        # 'encoder_hidden_size':[128,256,300,512],
        # 'mask_ratio': [0.5] ,#[x / 10 for x in range(0, 11, 2)],
        # "semi_supervised":[True,False]
    }

    # parameter_explore(opt, ps)  # super
    parameter_explore(opt.set({"semi_supervised": True}), ps,datasets=['laptop'])  # semi

    # main(opt_res.set({'valid_ratio': 0.5}))
    # main(opt_semi_res)
    # main(opt_res)
    # main(opt_lap)
    # main(opt_semi_lap.set({'mask_ratio': 0.7}))
