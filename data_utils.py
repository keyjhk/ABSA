import os, math, random
import pickle
import re
import string

import nltk
import numpy as np
import torch
import torch.nn.functional as F

import unicodedata
from torch.utils.data import Dataset

# dataset file
# the file format should be:
# row1: context,replace the aspect with $T$
# row2: aspect
# row3: polarity

LABELED_FILES = [
    'data/semeval14/Laptops_Test_Gold.xml.seg',
    'data/semeval14/Laptops_Train.xml.seg',
    'data/semeval14/Restaurants_Test_Gold.xml.seg',
    'data/semeval14/Restaurants_Train.xml.seg',
]

TRAIN_FILES = ['data/semeval14/Laptops_Train.xml.seg',
               'data/semeval14/Restaurants_Train.xml.seg',
               ]

UNLABELED_FILES = [
    'data/unlabeled/formated_electronic.txt',
    'data/unlabeled/formated_yelp_review.txt',

]

GLOVE_FILE = 'data/glove.42B.300d.txt'

# max seq len
MAX_SEQ_LEN = 85
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# token
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]  # PAD is the first,so its index is 0

# polarity
NEG_LABEL = 'NEG'
NEU_LABEL = 'NEU'
POS_LABEL = 'POS'

# pos
NOUN_LABEL = 'n'
VERB_LABEL = 'v'
ADVERB_LABEL = 'r'
ADJ_LABEL = 'a'
OTHER_LABEL = 'o'


class SentiWordNet:
    def __init__(self):
        # POS	ID	PosScore	NegScore	SynsetTerms	Gloss
        self.path = {
            'dict': 'data/SentiWordNet_3.0.0.txt',
            'save': 'state/sentiwordnet.pkl'
        }
        #
        self.senti_dict = {}
        self.load_dict()

    def __getitem__(self, item):
        return self.senti_dict.get(item, 0)

    def load_dict(self):
        if os.path.exists(self.path['save']):
            self.senti_dict = pickle.load(open(self.path['save'], 'rb'))
        else:
            self.process()
        print('senti_dict size:{}'.format(len(self.senti_dict)))

    def process(self):
        with open(self.path['dict'], 'r', encoding='utf8') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('#'): continue
                line = line.split('\t')
                if len(line) != 6: continue
                pos, pos_score, neg_score, synset_terms = line[0], line[2], line[3], line[4]
                wordsAndRank = synset_terms.split(' ')  # word#rank
                for wr in wordsAndRank:
                    word, rank = wr.split('#')
                    key = word + '#' + pos  # word#pos
                    try:
                        value = [rank, float(pos_score) - float(neg_score)]
                    except Exception:
                        print(pos_score, neg_score)
                    if self.senti_dict.get(key):
                        self.senti_dict[key].append(value)
                    else:
                        self.senti_dict[key] = [value]
        # calculate weighted score
        for key, value in self.senti_dict.items():
            # key:word#pos value:[[rank,score],..]
            # sum softmax(1/rank)*score
            scores, ranks = [], []
            for v in value:  # rank,score
                ranks.append(math.exp(1 / int(v[0])))  # weight: e^(1/rank)
                scores.append(v[1])
            ranks = [x / sum(ranks) for x in ranks]  # softmax
            score = sum(ranks[i] * scores[i] for i in range(len(ranks)))  # weighted_score
            # after update {word:score} positive if score>0 else negative
            self.senti_dict.update({key: score})

        pickle.dump(self.senti_dict, open(self.path['save'], 'wb'))


def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def pad_and_truncate(sequence, maxlen, value=0, dtype='int64'):
    x = (np.ones(maxlen) * value).astype(dtype)
    trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    x[:len(trunc)] = trunc  # trunc:pad
    return x


def build_tokenizer(max_seq_len, dat_fname='state/tokenizer.pkl',
                    mini_freq=1,
                    unlabeled=True):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        fnames = LABELED_FILES + UNLABELED_FILES if unlabeled else LABELED_FILES
        tokenizer = Tokenizer(max_seq_len, fnames=fnames, mini_freq=mini_freq)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))

    print('vocab size:{}'.format(len(tokenizer.word2idx)))

    return tokenizer  # 返回分词器


def build_embedding_matrix(word2idx, embed_dim=300, dat_fname='state/embedding_matatrix.pkl'):
    def _load_word_vec(path, embed_dim, word2idx, ):
        fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        word_vec = {}
        for line in fin:
            tokens = line.rstrip().split()
            word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
            if word in word2idx.keys():
                word_vec[word] = np.asarray(vec, dtype='float32')
        return word_vec

    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))
        fname = GLOVE_FILE
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        ctr = 0
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec  # 对于glove中的词 我们将其置为vec
                ctr += 1
        print('load {} words from glove'.format(ctr))
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def get_similar_tokens(query_token, embed_matrix, tokenizer, k=3):
    def knn(W, x, k):
        cos = F.cosine_similarity(W, x)
        _, topk = torch.topk(cos, k=k)
        topk = topk.cpu().numpy()
        return topk, [cos[i].item() for i in topk]

    num_words = embed_matrix.shape[0]
    qv = embed_matrix[tokenizer.word2idx[query_token]]  # embed_dim
    qv = qv.view(1, -1).expand(num_words, -1)
    topk, cos = knn(embed_matrix, qv, k + 1)
    topk = [tokenizer.idx2word[i] for i in topk[1:]]
    cos = cos[1:]
    # print(query_token)
    # for i, c in zip(topk, cos):  # 除去输入词
    #     print('cosine sim=%.3f: %s' % (c, i))
    return topk, cos


class Tokenizer(object):
    def __init__(self, max_seq_len, fnames, mini_freq):
        self.max_seq_len = max_seq_len
        self.sentidict = SentiWordNet()
        self.max_seq_len = max_seq_len
        self.mini_freq = mini_freq
        self.word_count = {}
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # POS vocab
        # verb noun adj adverb other
        self.idx2pos = [VERB_LABEL, NOUN_LABEL, ADJ_LABEL, ADVERB_LABEL, OTHER_LABEL]
        self.pos2idx = {x: idx for idx, x in enumerate(self.idx2pos)}

        # polar vocab
        self.idx2polar = [NEG_LABEL, NEU_LABEL, POS_LABEL]
        self.polar2idx = {x: idx for idx, x in enumerate(self.idx2polar)}

        # fit on text
        for fname in fnames:
            for text in self.read_text(fname):
                self.fit_on_text(text)
        # build
        self.build_vocab()

    def read_text(self, fname):
        with open(fname, 'r', encoding='utf-8') as f:
            ctr = 0
            while True:
                context = f.readline().rstrip()
                if context:
                    ctr += 1
                    text_left, _, text_right = [s.lower().strip() for s in context.partition("$T$")]
                    aspect = f.readline().rstrip().lower()  #
                    f.readline().rstrip()  # skip polarity/null
                    yield text_left + " " + aspect + " " + text_right
                else:
                    break
            print('read {} sentences from {}'.format(ctr, fname))

    def build_vocab(self):
        mini_freq = self.mini_freq
        tokens = filter(lambda k: self.word_count[k] >= mini_freq, self.word_count.keys())
        tokens = list(tokens)
        print('trimmed {} tokens'.format(len(self.word_count) - len(tokens)))
        tokens = SPECIAL_TOKENS + tokens
        # special tokens
        for token in tokens:
            self.word2idx[token] = self.idx
            self.idx2word[self.idx] = token
            self.idx += 1

    def fit_on_text(self, text):
        words = self.tokenize(text)
        for word in words:
            self.word_count[word] = self.word_count.get(word, 0) + 1
            # if word not in self.word2idx:
            #     self.word2idx[word] = self.idx
            #     self.idx2word[self.idx] = word
            #     self.idx += 1

    def tokenize(self, text, islower=True):
        return text.lower().split() if islower else text.split()

    def text_to_position(self, text_len, aspect_boundary):
        left_len = aspect_boundary[0]
        aspect_len = aspect_boundary[1] - aspect_boundary[0] + 1
        right_len = text_len - left_len - aspect_len

        left_seq = list(range(left_len, 0, -1))
        aspect_seq = [0] * aspect_len
        right_seq = list(range(1, right_len + 1))
        # 余下的填充为上限距离 表示不相关
        return pad_and_truncate(left_seq + aspect_seq + right_seq,
                                self.max_seq_len, value=self.max_seq_len)

    def text_to_pos_polar(self, text):
        '''
        verb: VB; noun: NN; adjecttive: JJ; adverb： VB; other： O
        '''
        tokens = self.tokenize(text, islower=False)
        pos_tags = nltk.pos_tag(tokens)  # token
        pos_seq, polar_seq = [], []
        for x in pos_tags:  # token,POS
            # only care adj and verb
            if x[1].startswith('VB'):
                pos = 'v'
            elif x[1].startswith('JJ'):
                pos = 'a'
            elif x[1].startswith('NN'):
                pos = 'n'
            elif x[1].startswith('RB'):
                pos = 'r'
            else:
                pos = 'o'

            word_polar = self.sentidict[x[0] + '#' + pos]
            if word_polar > 0:
                word_polar = POS_LABEL
            elif word_polar < 0:
                word_polar = NEG_LABEL
            else:
                word_polar = NEU_LABEL

            pos_seq.append(self.pos2idx[pos])
            polar_seq.append(self.polar2idx[word_polar])

        # 先pos 再polar
        return pad_and_truncate(pos_seq, self.max_seq_len, value=self.pos2idx[OTHER_LABEL]), \
               pad_and_truncate(polar_seq, self.max_seq_len, value=self.polar2idx[NEU_LABEL])

    def text_to_sequence(self, text, add_eos=False):
        words = self.tokenize(text)
        sequence = [self.word2idx[w] if w in self.word2idx else self.word2idx[UNK_TOKEN] for w in words]
        if add_eos:
            sequence.append(self.word2idx[EOS_TOKEN])
        # nparray : MAX_LENGTH
        return pad_and_truncate(sequence, self.max_seq_len, value=self.word2idx[PAD_TOKEN])

    def sequence_to_text(self, sequence, idx2char=None, skip_word=None):
        if not idx2char: idx2char = self.idx2word
        if not skip_word: skip_word = self.word2idx[PAD_TOKEN]
        return ' '.join(str(idx2char[idx]) for idx in sequence if idx != skip_word)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer, write_file=False, combine=False):
        # data
        self.all_data = {}  #
        self.data = []  # for iterater
        self.statistic_data = {
            'sentences': 0,
            'avg_length': 0,
            "aspect": set(),
            "polar_count": None  # {pos:int,neg:int,neu:int}
        }
        # dataset name
        self.dataset_name = os.path.basename(fname)  # get file name
        self.dat_fname = 'state/absa_dataset_{}.pkl'.format(self.dataset_name)  # for save/load
        # tokenize
        self.tokenizer = tokenizer
        self.combine = combine  # 是否合并一句话里的多个aspect

        if os.path.exists(self.dat_fname):
            self.load_dataset()
        else:
            self.build_alldata(fname)
            self.build_dataset()
            self.statistic()

            self.save_dataset()

        if write_file: self.write_formarted_datafile()
        self.show_dataset()

    def show_dataset(self):
        print('dataset:[{}] \nsentences:{} avg_len:{} aspects:{} polars:[{}]\n'.format(
            self.dataset_name,
            self.statistic_data['sentences'],
            self.statistic_data['avg_length'],
            len(self.statistic_data['aspect']),
            self.statistic_data['polar_count']))

    def build_alldata(self, fname):
        '''
        alldata:{
            context:{
                context_indices:,
                text_indices:,
                ....
            }
        }
        :param fname:数据集文件
        :return: alldata
        '''
        all_data = {}
        tokenizer = self.tokenizer
        combine = self.combine
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            while True:
                context = f.readline().rstrip()
                if context:
                    text_left, _, text_right = [s.strip() for s in context.partition("$T$")]
                    aspect = f.readline().rstrip()
                    polarity = f.readline().rstrip()

                    context = text_left + " " + aspect + " " + text_right

                    # text(no aspect),context(text with aspect)
                    add_eos = combine  # 当需要组合的时候，末尾添加eos_token
                    pad_idx = self.tokenizer.word2idx[PAD_TOKEN]

                    # text to indices with same length(max_lemn),numpy ndarray
                    text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right,
                                                              add_eos=add_eos)
                    context_indices = tokenizer.text_to_sequence(context, add_eos=add_eos)
                    context_len = np.sum(context_indices != pad_idx)
                    left_indices = tokenizer.text_to_sequence(text_left)
                    right_indices = tokenizer.text_to_sequence(text_right)
                    aspect_indices = tokenizer.text_to_sequence(aspect)

                    aspect_len = len(tokenizer.tokenize(aspect))
                    left_len = len(tokenizer.tokenize(text_left))
                    aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
                    polarity = int(polarity) + 1 if polarity != '' else -1  # neg:0 neu:1 pos:2 null:-1(unlabeled)
                    pos_indices, polar_indices = tokenizer.text_to_pos_polar(context)  # part of speech/polar
                    position_indices = tokenizer.text_to_position(context_len, aspect_boundary)

                    if all_data.get(context):
                        # context aspect
                        all_data[context]['text_indices'].append(text_indices)
                        all_data[context]['left_aspect_right_indices'].append(
                            (left_indices, aspect_indices, right_indices))
                        all_data[context]['aspect_boundary'].append(aspect_boundary)
                        all_data[context]['aspect_indices'].append(aspect_indices)
                        all_data[context]['position_indices'].append(position_indices)
                        all_data[context]['polarity'].append(polarity)
                    else:
                        # 一句话里可能有多个属性
                        all_data[context] = {
                            'text_indices': [text_indices],
                            'context_indices': context_indices,
                            'context_len': context_len,
                            'pos_indices': pos_indices,
                            'polar_indices': polar_indices,
                            'position_indices': [position_indices],
                            'aspect_indices': [aspect_indices],
                            'left_aspect_right_indices': [(left_indices, aspect_indices, right_indices)],
                            'aspect_boundary': [aspect_boundary],
                            'polarity': [polarity],
                        }


                else:
                    break
        self.all_data = all_data

    def build_dataset(self):
        all_data = self.all_data
        combine = self.combine
        max_seq_len = self.tokenizer.max_seq_len
        pad_token_idx = self.tokenizer.word2idx[PAD_TOKEN]
        for context, val in all_data.items():
            # meta struct
            data_meta = {
                'context_indices': val['context_indices'],
                'pos_indices': val['pos_indices'],
                'polar_indices': val['polar_indices'],
                'position_indices': 0,
                'aspect_indices': 0,
                'aspect_boundary': 0,
                'target': 0,
                'len_s': val['context_len'],
                'len_t': 0,
                'mask_s': val['context_indices'] != pad_token_idx,  # mask for src
                'mask_t': 0  # mask for target
            }
            if combine:
                data_item = data_meta.copy()
                target = []  # (as,ae,p),..eos_position
                for i in range(len(val['polarity'])):
                    target.append(val['aspect_boundary'][i][0])  # as
                    target.append(val['aspect_boundary'][i][1])  # ae
                    target.append(val['polarity'][i])  # p
                target.append(val['context_len'] - 1)  # eos position in context

                data_item['len_t'] = len(target)

                mask_t = np.zeros(max_seq_len, dtype=np.bool)
                mask_t[:len(target)] = True
                data_item['mask_t'] = mask_t
                data_item['target'] = pad_and_truncate(target, max_seq_len)

                self.data.append(data_item)
            else:
                # 1 context 1 aspect
                for i in range(len(val['polarity'])):
                    data_item = data_meta.copy()
                    data_item['target'] = val['polarity'][i]
                    data_item['text_indices'] = val['text_indices'][i]
                    data_item['position_indices'] = val['position_indices'][i]
                    data_item['aspect_indices'] = val['aspect_indices'][i]
                    data_item['aspect_boundary'] = val['aspect_boundary'][i]
                    self.data.append(data_item)

    def load_dataset(self):
        print('loading dataset:[{}]'.format(self.dat_fname))
        datas = pickle.load(open(self.dat_fname, 'rb'))
        self.data = datas['data']
        self.all_data = datas['all_data']
        self.statistic_data = datas['statistic']

    def save_dataset(self):
        datas = {
            'data': self.data,
            'all_data': self.all_data,
            'statistic': self.statistic_data
        }
        pickle.dump(datas, open(self.dat_fname, 'wb'))

    def write_formarted_datafile(self):
        data = self.all_data
        tokenizer = self.tokenizer
        new_file = 'state/formated_{}.txt'.format(self.dataset_name)
        print('writing {}……'.format(new_file))

        with open(new_file, 'w', encoding='utf8') as f:
            for x, y in data.items():
                content_len = y['context_len']
                context = tokenizer.sequence_to_text(y['context_indices'][:content_len], tokenizer.idx2word)
                pos = tokenizer.sequence_to_text(y['pos_indices'][:content_len], tokenizer.idx2pos)
                polar = tokenizer.sequence_to_text(y['polar_indices'][:content_len], tokenizer.idx2polar)
                ct_pos_polar = '\n'.join([context, pos, polar, ''])

                t = ''
                for i in range(len(y['polarity'])):
                    sidx, eidx = y['aspect_boundary'][i][0], y['aspect_boundary'][i][1]
                    aspect = ' '.join(context.split()[sidx:eidx + 1])
                    polarity = y["polarity"][i]
                    t += "{} {},{} {}\t\t".format(aspect, sidx, eidx, polarity)

                t += str(content_len)
                f.write(ct_pos_polar)
                f.write(t + '\n' * 2)

        with open('state/formated_datafile.txt', 'w', encoding='utf8') as f:
            for x, y in data.items():
                content_len = y['context_len']
                f.write(tokenizer.sequence_to_text(y['context_indices'][:content_len], tokenizer.idx2word) + '\n')
                f.write(tokenizer.sequence_to_text(y['pos_indices'][:content_len], tokenizer.idx2pos) + '\n')
                f.write(tokenizer.sequence_to_text(y['polar_indices'][:content_len], tokenizer.idx2polar) + '\n')
                for i in range(len(y['polarity'])):
                    f.write(tokenizer.sequence_to_text(y['position_indices'][i][:content_len],
                                                       list(range(tokenizer.max_seq_len))) + '\n')

                t = ''
                for i in range(len(y['polarity'])):
                    sidx, eidx = y['aspect_boundary'][i][0], y['aspect_boundary'][i][1]
                    aspect = ' '.join(x.split()[sidx:eidx + 1])
                    polarity = y["polarity"][i]
                    t += aspect + ' ' + str(sidx) + ',' + str(eidx) + ' ' + str(polarity) + '\t'
                t += str(content_len - 1)
                f.write(t + '\n' * 2)

    def statistic(self):
        '''
        all_data[context] = {
                        'text_indices': [np.array],
                        'context_indices': np.array,
                        'context_len': int,
                        'pos_indices': np.array,
                        'polar_indices': np.array,
                        'left_aspect_right_indices': [(left_indices, aspect_indices, right_indices)],
                        'aspect_boundary': [np.array[start,end],..],
                        'polarity': [polarity(0/1),..],
                    }
        :param all_data:
        :return:
        '''

        tokenizer = self.tokenizer
        data = self.all_data

        sentences = 0
        sent_length = 0
        aspects = set()
        polar_count = {}.fromkeys(tokenizer.polar2idx.keys(), 0)

        for context, val in data.items():
            words = context.split(' ')
            for i in range(len(val['polarity'])):
                aspect_start, aspect_end = val['aspect_boundary'][i]
                aspect = ' '.join(words[aspect_start:aspect_end + 1])
                aspects.add(aspect)
                polarity = val['polarity'][i]
                if polarity != -1:  # labeled
                    polar = tokenizer.idx2polar[polarity]
                    polar_count[polar] = polar_count.get(polar) + 1

            sentences += 1
            sent_length += len(words)

        self.statistic_data['sentences'] = sentences
        self.statistic_data['avg_length'] = sent_length // sentences
        self.statistic_data['aspect'] = aspects
        self.statistic_data['polar_count'] = polar_count

    def union(self, dataset):
        print('union [{}] and [{}]'.format(self.dataset_name, dataset.dataset_name))
        self.dataset_name = "{}_{}".format(self.dataset_name, dataset.dataset_name)
        self.all_data.update(dataset.all_data)
        self.data.extend(dataset.data)
        self.statistic()
        self.show_dataset()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MixDataLoader:
    def __init__(self, labeled_loader, unlabeld_loader, eda_loader=None, semi_supervised=True):
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeld_loader
        self.eda_loader = eda_loader
        self.semi_supervised = semi_supervised
        self.label_len = len(labeled_loader)
        self.unlabel_len = len(unlabeld_loader)

    def _endless_labeled(self):
        while True:
            for batch in self.labeled_loader:
                yield batch
            if self.eda_loader is not None:
                for batch in self.eda_loader:
                    yield batch

    def alternating_batch(self):
        labeled_loader = self._endless_labeled()  # generator
        while True:
            for u_batch in self.unlabeled_loader:
                yield next(labeled_loader), 'labeled'
                if self.semi_supervised:
                    yield u_batch, 'unlabeled'


class DataAug:
    def __init__(self, fnames, tokenizer, embed_matrix):
        self.fnames = fnames
        self.tokenizer = tokenizer
        self.embed = torch.tensor(embed_matrix, device=DEVICE)
        self.window = 3
        self.datas = []

    def augmentation(self):
        for fname in self.fnames:
            print('aug {}……'.format(fname))
            datas = []
            with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
                while True:
                    context = f.readline().rstrip()
                    if not context: break
                    text_left, _, text_right = [s.strip() for s in context.partition("$T$")]
                    aspect = f.readline().rstrip()
                    polarity = f.readline().rstrip()

                    context = text_left + " " + aspect + " " + text_right
                    len_left = len(self.tokenizer.tokenize(text_left))
                    len_aspect = len(self.tokenizer.tokenize(aspect))
                    len_context = len(self.tokenizer.tokenize(context))
                    position_l = max(0, len_left - self.window)
                    position_r = min(len_left + len_aspect + self.window, len_context - 1)

                    positions = list(range(0, position_l)) + list(range(position_r, len_context))
                    new_sentences = self._augmentation(context, positions)
                    for s in new_sentences:
                        datas.append((s, aspect.lower(), polarity))

            self.write_file(datas, fname)

    def _augmentation(self, sentence, positions):
        edas = [self.replace, self.delete, self.add, self.swap]
        tokens = self.tokenizer.tokenize(sentence)
        new_sentences = []
        for eda in edas:
            position = random.choice(positions)
            new_sentence = eda(tokens[:], position)  # copy
            # print(new_sentence)
            new_sentences.append(new_sentence)

        return new_sentences

    def write_file(self, datas, fname):
        # datas[(sentence,aspect,polarity)]
        fname = os.path.basename(fname)
        fname = os.path.join('state', 'eda_%s' % fname)
        print('write {} sentences:{}'.format(fname, len(datas)))
        with open(fname, 'w', encoding='utf8') as f:
            for sentence, aspect, polarity in datas:
                sentence = sentence.partition(aspect)
                sentence = sentence[0] + "$T$" + sentence[2]

                f.write(sentence + '\n')
                f.write(aspect + '\n')
                f.write(str(polarity) + '\n')

    def replace(self, tokens, p):
        token = tokens[p]
        new_tokens, _ = get_similar_tokens(token, embed_matrix=self.embed, tokenizer=self.tokenizer)
        new_token = new_tokens[random.randint(0, len(new_tokens) - 1)]
        tokens[p] = new_token
        # print('REPLACE [{}] with [{}]'.format(tokens[p], new_token))
        return ' '.join(tokens)

    def add(self, tokens, position):
        num_words = len(self.tokenizer.word2idx)
        new_token = random.randint(len(SPECIAL_TOKENS), num_words - 1)
        new_token = self.tokenizer.idx2word[new_token]
        tokens.insert(position, new_token)
        # print('ADD [{}] in [{}]'.format(new_token, position))
        return ' '.join(tokens)

    def swap(self, tokens, position):
        length = len(tokens)
        if length > 2:
            position1 = position + 1 if position + 1 < length else position - 1
            # print('SWAP [{}], [{}]'.format(tokens[position], tokens[position1]))
            tokens[position], tokens[position1] = tokens[position1], tokens[position]
        return ' '.join(tokens)

    def delete(self, tokens, position):
        # print('DELETE [{}] at [{}]'.format(tokens[position], position))
        tokens.pop(position)
        return ' '.join(tokens)


if __name__ == '__main__':
    # if the tokenizer rebuild ,the embedding matrix should rebuilf too
    # because the word:idx map has changed ,but the embeding matrix may
    # reload from the disk
    tokenizer = build_tokenizer(max_seq_len=MAX_SEQ_LEN)
    embed_matrix = build_embedding_matrix(tokenizer.word2idx)

    # augmentation
    # da = DataAug(fnames=TRAIN_FILES, embed_matrix=embed_matrix, tokenizer=tokenizer)
    # da.augmentation()

    # labeled
    # ABSADataset(fname='data/semeval14/Laptops_Train.xml.seg', tokenizer=tokenizer)
    # ABSADataset(fname='data/semeval14/Laptops_Test_Gold.xml.seg', tokenizer=tokenizer)
    # ABSADataset(fname='data/semeval14/Restaurants_Train.xml.seg', tokenizer=tokenizer)
    # ABSADataset(fname='data/semeval14/Restaurants_Test_Gold.xml.seg', tokenizer=tokenizer)
    # unlabeled
    # ABSADataset(fname='data/unlabeled/formated_electronic.txt', tokenizer=tokenizer)
    # ABSADataset(fname='data/unlabeled/formated_yelp_review.txt', tokenizer=tokenizer)
    # eda
    # eda_lap=ABSADataset(fname='data/eda/eda_Laptops_Train.xml.seg', tokenizer=tokenizer)
    # eda_res=ABSADataset(fname='data/eda/eda_Restaurants_Train.xml.seg', tokenizer=tokenizer)
