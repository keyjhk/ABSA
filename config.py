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
    'print_every': 1,
    'step_every': 0.25,  # 0<x<1 ,show progress when achieved
    'patience': 10,  # how many epoches does't increase then stop train
    'semi_patience': 10,  # 10 speed for test
    'batch_size': 64,
    'semi_supervised': False,
    'clear_model': True,  # clear saved models before run ,prevent load
    'max_seq_len': 85,
    'valid_ratio': 0.1,
    'dataset': 'laptop',
    "save_model_name": '{dataset}_{model}_epoch{epoch}_acc_{acc:.2f}_f1_{f1:.2f}.pkl',
    # eval
    'eval_times': 3,
    # reproduce
    'seed': 544,
    # optimizer
    'semi_lr': 5e-4,  # 5e-4,
    'semi_l2': 5e-3,  # 5e-3,
    'lr': 1e-3,  # 1e-3,
    'l2': 5e-3,  # 1e-2,
    # reset:
    "initializer": "xavier_uniform_",
    # model
    "word_embedding_size": 300,
    "pos_embedding_size": 50,
    "polar_embedding_size": 50,
    "position_embedding_size": 50,
    "encoder_hidden_size": 300,
    # dynamic mask/weight
    "threshould": 8,  # 26 lap,12 res
    'drop_attention': 0.5,
    'mask_ratio': 1,
    'weight_alpha': 1,
    'weight_keep': True,
    # cvt
    'drop_lab': 0.5,
    'drop_unlab': 0.6,
    'unlabeled_loss': 'mask_strong',  # 'mask_weak', mask_window  # mask_strong ,all,  weight
    'loss_alpha': 1,
    'loss_cal': 'kl'  # loss

}

# not show in log
EXCLUDE = ['datasets', 'print_every', 'step_every', 'clear_model', 'initializers',
           'patience', 'semi_patience', 'max_seq_len', 'save_model_name']
