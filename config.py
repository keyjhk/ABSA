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
    'gpu_parallel': False,
    'device_ids': [7, 6, 5, 4],
    'print_every': 1,
    'step_every': 0.5,  # 0<x<1 ,show progress when achieved
    'patience': 10,  # early stop
    'batch_size': 32,
    'semi_supervised': False,
    'clear_model': True,  # clear saved models before run ,retrain ,prevent load
    'max_seq_len': 85,
    'valid_ratio': 0,
    'dataset': 'laptop',
    "save_model_name": '{dataset}_{model}_epoch{epoch}_acc_{acc:.2f}_f1_{f1:.2f}.pkl',
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
    "encoder_hidden_size": 300,
    # dynamic mask/weight
    'drop_attention': 0,
    "threshould": 8,  # 26 lap,12 res
    'mask_ratio': 0.4,
    # cvt
    'drop_lab': 0.2,
    'drop_unlab': 0.5,
    'unlabeled_loss': 'mask_strong',  # 'mask_weak', mask_window  # mask_strong ,all,  weight
    'use_weight':False,

}

# not show in log
EXCLUDE = ['datasets', 'print_every', 'step_every', 'clear_model', 'initializers',
           'patience', 'semi_patience', 'max_seq_len', 'save_model_name']
