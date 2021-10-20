import os
import gzip, json
from nltk import word_tokenize


def parse(fname, dataset='restaurant'):
    g = gzip.open(fname, 'r')
    for i, l in enumerate(g):
        if dataset == 'laptop':
            yield json.loads(l)['reviewText']
        elif dataset == 'restaurant':
            if i == 0: continue
            yield json.loads(l)['text']


def main(dataset):
    max_seqlen = 85  # max_len of a sentence
    total = 5e4  # how many sentences

    path = {
        'laptop': 'data/unlabeled/Electronics_5.json.gz',
        'restaurant': 'data/unlabeled/yelp_review.json.tar.gz'
    }

    file_name = path[dataset]
    new_file_name = 'state/{}.txt'.format(os.path.basename(file_name).split('.')[0])

    with open(new_file_name, 'w', encoding='utf8') as f:
        for i, l in enumerate(parse(file_name, dataset=dataset)):
            if i > total: break
            text = l.rstrip()
            text = ' '.join(word_tokenize(text)[:max_seqlen])
            if not text: continue
            # print(text)
            f.write(text + '\n' * 2)


if __name__ == '__main__':
    dataset = 'restaurant'
    # dataset = 'laptop'
    main(dataset)
