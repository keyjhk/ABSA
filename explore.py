# explore optim hyparameters
from time import strftime, localtime
import random, sys
from train import set_logger, Instructor
from config import DEFAULT_OPTION


def parameter_explore(opt, par_vals, datasets=None, semi_sup_compare=False):
    # par_vals:{parameter_name:[parameter_value,...],...}
    def is_valid_option(opt):
        is_valid_opt = True
        logger.info('comparing……'.center(30, '='))

        # valid_ratios = [0, 0.25, 0.5, 0.75]
        ratio_name = 'train_len'
        ratios = [None, 1500, 1000, 500]
        # only these factors influence supervised
        valid_opt_name = '{lr}_{l2}_{window_weight}_{drop_lab}'.format(lr=opt.lr, l2=opt.l2,
                                                                       window_weight=opt.window_weight,
                                                                       drop_lab=opt.drop_lab)
        # {valid_opt_name:{ratio:acc,...}}
        if not semi_valid_ratio_scores.get(valid_opt_name):  # init
            semi_valid_ratio_scores[valid_opt_name] = {x: None for x in ratios}
        semi_scores = semi_valid_ratio_scores.get(valid_opt_name)  # {ratio:acc}

        for ratio in ratios:
            # the difference between sup/sems is only the toggle of semi_supervised
            # keep other settings same:drop_lab/drop_unlab/mask_ratio
            # sup
            _opt = opt.set({ratio_name: ratio}, name='compare_' + opt.name)

            if not semi_scores.get(ratio):
                _ins_sup = Instructor(_opt.set({'semi_supervised': False}), logout=False)
                res_sup = _ins_sup.run()
                semi_scores[ratio] = res_sup
            else:
                res_sup = semi_scores.get(ratio)  # fetch sup results directly

            # semi
            _ins_semi = Instructor(_opt.set({'semi_supervised': True}), logout=False)
            res_semi = _ins_semi.run()
            logger.info(
                'ratio: {} semi[ acc:{} f1:{} ] super_acc[ acc:{} f1:{} ]'.format(ratio, res_semi['acc'],
                                                                                  res_semi['f1'],
                                                                                  res_sup['acc'], res_sup['f1']))

            is_valid_opt = res_semi['acc'] > res_sup['acc']
            if not is_valid_opt:
                logger.info('option:{} is dropped!'.format(opt.name).center(30, '='))
                break
        else:
            logger.info('option:{} is valid!'.format(opt.name).center(30, '='))
        return is_valid_opt

    datasets = ['laptop'] if datasets is None else datasets

    # logger
    logger_fname = 'p'
    for p in par_vals.keys():
        logger_fname += '_{}{}'.format(p, len(par_vals[p]))  # p_{parameter_name}{search_len}
    logger = set_logger(name='parameter_explore',
                        file='{}_{}.log'.format(logger_fname, strftime("%m%d-%H%M", localtime())))

    # dynamic build search_options
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

    # filter invalid options,when drop_unlab <= drop_lab
    search_options = filter(lambda option: option.drop_lab <= option.drop_unlab, search_options)
    if semi_sup_compare:
        semi_valid_ratio_scores = {}  # saved different scores with same options
        logger.warning('have opened the semi_sup_compare!!!')
    search_results = {d: [] for d in datasets}

    # search
    best_result = 0
    best_params = None
    for dataset in datasets:
        results = []
        for search_option in search_options:
            if semi_sup_compare and not is_valid_option(search_option.set({'dataset': dataset})):
                continue
            _ins = Instructor(search_option.set({'dataset': dataset}))
            res = _ins.run()
            results.append(res)
            vr = '[dataset]:{dataset} [{option}] [result]:{result}'.format(dataset=dataset,
                                                                           option=search_option.name,
                                                                           result=res)
            search_results[dataset].append(vr)
            logger.info(vr)
            acc = res['acc']
            if acc > best_result:
                best_result = acc
                best_params = vr

    # show results
    logger.info('final results'.center(30, '*'))
    logger.info('sys:{}'.format(sys.platform, ))
    for d, res in search_results.items():
        # d:dataset res:List[Str]
        res = sorted(res) if len(par_vals) > 1 else res  # random search ,then sort the results
        for r in res: logger.info(r)
    logger.info('*' * 30)
    logger.info('best params：{}'.format(best_params))


if __name__ == '__main__':
    opt = DEFAULT_OPTION
    ps = {
        # 'batch_size': [32, 64],
        # 'lr': [1e-2, 1e-3, 1e-4],
        # 'l2': [5e-3, 1e-3, 5e-4,1e-4,5e-5,1e-5],
        # 'encoder_hidden_size':[300,512,768,1024]
        # 'window_weight': range(1,10,2),
        # 'drop_lab': [x / 10 for x in range(1, 6)],
        # 'drop_unlab': [x / 10 for x in range(3, 9)],
        # 'unlabel_len': [5000, 10000,15000,20000],
        # 'train_len': [500, 1000,1500, None],
        # 'semi_supervised': [True, False],
        # 'use_weight': [False, True]
    }

    datasets = opt.datasets.keys()
    # parameter_explore(opt, ps)  # super default lap
    # parameter_explore(opt, ps , datasets=datasets)  # super all
    # parameter_explore(opt, ps, datasets=['restaurant'])  # restaurant
    #
    # parameter_explore(opt.set({"semi_supervised": True}), ps,
    #                   semi_sup_compare=True,
    #                   datasets=['laptop'])  # semi default laptop restaurant
    # parameter_explore(opt.set({"semi_supervised": True}), ps)  # semi default lap
    parameter_explore(opt.set({"semi_supervised": True,}), ps,datasets=['restaurant'])  # semi default res
    # parameter_explore(opt.set({"ssemi_supervised": True}), ps,datasets=datasets)  # semi all#
