import torch

PARAMETERS = {
    # datasets
    'datasets': {
        'restaurant': {
            'train': 'data/semeval14/Restaurants_Train.xml.seg',
            'test': 'data/semeval14/Restaurants_Test_Gold.xml.seg',
            'unlabeled': 'data/unlabeled/formated_yelp_review.txt'
        },
        'laptop': {
            'train': 'data/semeval14/Laptops_Train.xml.seg',
            'test': 'data/semeval14/Laptops_Test_Gold.xml.seg',
            'unlabeled': 'data/unlabeled/formated_electronic.txt'
        }
    },
    # train
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'device_ids': [6, 5, 7, 0],
    'print_every': 1,
    'step_every': 0.5,  # 0<x<1 ,show progress when achieved
    'patience': 10,  # early stop
    'batch_size': 32,
    'semi_supervised': False,
    'clear_model': True,  # clear saved models each run
    'max_seq_len': 85,
    'valid_ratio': 0,
    'train_len': None,
    'unlabel_len': 2e4,
    'dataset': 'laptop',
    "save_model_name": '{dataset}_{time}_{model}_epoch{epoch}_acc_{acc:.2f}_f1_{f1:.2f}.pkl',
    # eval
    'eval_times': 3,
    # reproduce
    'seed': 544,
    # optimizer
    'lr': 1e-3,
    'l2': 5e-3,
    # reset:
    "initializer": "xavier_uniform_",
    # model
    "word_embedding_size": 300,
    "pos_embedding_size": 50,
    "polar_embedding_size": 50,
    "position_embedding_size": 50,
    "encoder_hidden_size": 300,  # 300
    # dynamic mask/weight
    'window_weight': 2,  # res 0,lap 2
    "window_mask": None,
    'mask_ratio': None,
    # cvt
    # res: 0.4 0.5  ; lap: 0.2 0.5
    'drop_lab': 0.2,
    'drop_unlab': 0.5,

}

# not show in log
EXCLUDE = ['datasets', 'print_every', 'step_every', 'clear_model', 'initializers',
           'patience', 'semi_patience', 'max_seq_len', 'save_model_name']


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


DEFAULT_OPTION = Option(PARAMETERS)
