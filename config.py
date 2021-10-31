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
    'step_every': 0.1,  # 0<x<1 ,show progress when achieved
    'patience': 10,  # how many epoches does't increase then stop train
    'semi_patience': 30,
    'batch_size': 32,
    'semi_supervised': False,
    'clear_model': True,  # clear saved models before run ,prevent load
    'max_seq_len': 85,
    'valid_ratio': 0,
    'dataset': 'laptop',
    "save_model_name": '{dataset}_{model}_epoch{epoch}_acc_{acc:.2f}_f1_{f1:.2f}.pkl',
    # eval
    'eval_times': 10,
    # optimzer
    'semi_lr': 1e-3,
    'semi_l2': 5e-3,
    'lr': 1e-3,
    'l2': 1e-2,
    # reset:
    "initializer": "xavier_uniform_",
    # model
    "word_embedding_size": 300,
    "pos_embedding_size": 50,
    "polar_embedding_size": 50,
    "position_embedding_size": 50,
    "encoder_hidden_size": 300,
    # dynamic mask/weight
    "threshould": 4,
    'mask_ratio': 0.5,
    'weight_keep': False

}

# not show in log
EXCLUDE = ['datasets', 'print_every', 'step_every', 'clear_model', 'initializers',
           'patience', 'semi_patience', 'max_seq_len', 'save_model_name']
