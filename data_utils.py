import os, math, pickle
import string

import nltk
import numpy as np

import unicodedata
from torch.utils.data import Dataset

# token
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

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

        self.senti_dict = {}
        self.load_dict()

    def __getitem__(self, item):
        return self.senti_dict.get(item, 0)

    def load_dict(self):
        if os.path.exists(self.path['save']):
            self.senti_dict = pickle.load(open(self.path['save'], 'rb'))
        else:
            self.process()
        print('dict_len:{}'.format(len(self.senti_dict)))

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
            for v in value:
                ranks.append(math.exp(1 / int(v[0])))
                scores.append(v[1])
            ranks = [x / sum(ranks) for x in ranks]  # softmax
            score = sum(ranks[i] * scores[i] for i in range(len(ranks)))  # weighted_score
            # after update {word:score} positive if score>0 else negative
            self.senti_dict.update({key: score})

        pickle.dump(self.senti_dict, open(self.path['save'], 'wb'))


def test():
    '''
    verb: VB
    noun: NN
    adjecttive: JJ
    adverb： VB
    other： O
    :return:
    '''
    print(nltk.pos_tag('I charge it at night'.split()))

    from torchtext.data.utils import get_tokenizer

    tokenizer = get_tokenizer('basic_english')  # 分词器

    # 分词器输入 句子 ，返回 token序列
    print(nltk.pos_tag(tokenizer("it's a boy")))  # [i,come,from,china]


def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def pad_and_truncate(sequence, maxlen, value, dtype='int64'):
    x = (np.ones(maxlen) * value).astype(dtype)
    trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    x[:len(trunc)] = trunc  # trunc:pad
    return x


def build_tokenizer(fnames, max_seq_len, dat_fname='state/tokenizer.pkl'):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        tokenizer = Tokenizer(max_seq_len, fname=fnames)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))

    return tokenizer  # 返回分词器


def build_embedding_matrix(word2idx, embed_dim=300, dat_fname='state/embedding_matatrix.pkl'):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))
        fname = 'data/glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec  # 对于glove中的词 我们将其置为vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


class Tokenizer(object):
    def __init__(self, max_seq_len, fname=None,
                 special_tokens=[PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]):
        self.max_seq_len = max_seq_len
        self.sentidict = SentiWordNet()
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # special tokens
        if special_tokens:
            for token in special_tokens:
                self.word2idx[token] = self.idx
                self.idx2word[self.idx] = token
                self.idx += 1

        # POS vocab
        # verb noun adj adverb other
        self.idx2pos = [VERB_LABEL, NOUN_LABEL, ADJ_LABEL, ADVERB_LABEL, OTHER_LABEL]
        self.pos2idx = {x: idx for idx, x in enumerate(self.idx2pos)}

        # polar vocab
        self.idx2polar = [NEG_LABEL, NEU_LABEL, POS_LABEL]
        self.polar2idx = {x: idx for idx, x in enumerate(self.idx2polar)}

        if fname:
            self.fit_on_text(self.read_text(fname))

    def read_text(self, fnanme):
        if os.path.isdir(fnanme):
            files = []
            for f in os.listdir(fnanme):
                files.append(os.path.join(fnanme, f))
        else:
            files = [fnanme]

        text = ''
        for f in files:
            fin = open(f, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    def fit_on_text(self, text):
        words = self.tokenize(text)
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def tokenize(self, text, islower=True):
        return text.lower().split() if islower else text.split()

    def text_to_pos_polar(self, text):
        '''
        verb: VB; noun: NN; adjecttive: JJ; adverb： VB; other： O
        '''
        tokens = self.tokenize(text, islower=False)
        pos_tags = nltk.pos_tag(tokens)  # token
        pos_seq, polar_seq = [], []
        for x in pos_tags:  # token,POS
            if x[1].startswith('VB'):
                pos = 'v'
            elif x[1].startswith('NN'):
                pos = 'n'
            elif x[1].startswith('JJ'):
                pos = 'a'
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

    def sequence_to_text(self, sequence, idx2char, skip_word=None):
        return ' '.join(idx2char[idx] for idx in sequence if idx != skip_word)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer, write_file=False, dat_fname='state/absa_dataset_', combine=True):
        self.data = []
        self.dataset_name = os.path.basename(fname)
        self.tokenizer = tokenizer
        self.combine = combine  # 是否合并一句话里的多个aspect

        dat_fname = dat_fname + self.dataset_name + '.pkl'

        if os.path.exists(dat_fname):
            print('loading absa_dataset:', dat_fname)
            self.data = pickle.load(open(dat_fname, 'rb'))
        else:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()

            all_data = {}
            for i in range(0, len(lines), 3):
                # 一开始不能小写 会影响 pos词性判断
                text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].strip()
                polarity = lines[i + 2].strip()
                context = text_left + " " + aspect + " " + text_right

                # text(no aspect),context(text with aspect)
                add_eos = combine  # 当需要组合的时候，末尾添加eos_token
                text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right, add_eos=add_eos)
                context_indices = tokenizer.text_to_sequence(context, add_eos=add_eos)
                context_len = len(tokenizer.tokenize(context)) + 1 if add_eos else len(
                    tokenizer.tokenize(context))  # 原句子长度+1 eos
                left_indices = tokenizer.text_to_sequence(text_left)
                right_indices = tokenizer.text_to_sequence(text_right)
                aspect_indices = tokenizer.text_to_sequence(aspect)

                aspect_len = len(tokenizer.tokenize(aspect))
                left_len = len(tokenizer.tokenize(text_left))
                aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
                polarity = int(polarity) + 1  # neg:0 neu:1 pos:2
                pos_indices, polar_indices = tokenizer.text_to_pos_polar(context)

                if all_data.get(context):
                    # context aspect
                    all_data[context]['text_indices'].append(text_indices)
                    all_data[context]['left_aspect_right_indices'].append((left_indices, aspect_indices, right_indices))
                    all_data[context]['aspect_boundary'].append(aspect_boundary)
                    all_data[context]['polarity'].append(polarity)
                else:
                    # 一句话里可能有多个属性
                    all_data[context] = {
                        'text_indices': [text_indices],
                        'context_indices': context_indices,
                        'context_len': context_len,
                        'pos_indices': pos_indices,
                        'polar_indices': polar_indices,
                        'left_aspect_right_indices': [(left_indices, aspect_indices, right_indices)],
                        'aspect_boundary': [aspect_boundary],
                        'polarity': [polarity],
                    }

            # formatted file
            if write_file:
                self.write_formarted_datafile(all_data)

            pickle.dump(self.data, open(dat_fname, 'wb'))

        self.statistic()

    def build_dataset(self, all_data):
        combine = self.combine
        if combine:
            # format data to src,target
            # seq,pos,polar : ae,as,p,...,eos_position
            for x, y in all_data.items():
                src = [y['context_indices'], y['pos_indices'], y['polar_indices']]  # content,pos,polar
                target = []  # (as,ae,p),..eos_position
                for i in range(len(y['polarity'])):
                    target.append(y['aspect_boundary'][i][0])  # as
                    target.append(y['aspect_boundary'][i][1])  # ae
                    target.append(y['polarity'][i])  # p
                target.append(y['context_len'] - 1)  # eos position

                len_x = y['context_len']
                len_y = len(target)
                # 对齐target
                target = pad_and_truncate(target, tokenizer.max_seq_len, 0)  # 用0填充 最后用mask标记即可
                # mask_s for pad_token(don't need attention) ;mask_t for not pad_token(calculate_loss)
                mask_s = y['context_indices'] == tokenizer.word2idx[PAD_TOKEN]
                mask_t = target != -1
                self.data.append((src, target, len_x, len_y, mask_s, mask_t))
        else:
            # format data to src,target:0
            # src:(seq,pos,polar,apect) target:polarity of apsect
            for x, y in all_data.items():
                src, target = [], []
                for i in range(len(y['polarity'])):
                    # content,pos,polar,aspect_boundary
                    src.extend([y['context_indices'],
                                y['pos_indices'],
                                y['polar_indices'],
                                y['aspect_boundary'][i]])

                    target.append(y['polarity'][i])  # p
                    self.data.append(src, target)

    def write_formarted_datafile(self, data):
        tokenizer = self.tokenizer
        with open('state/formated_datafile.txt', 'w', encoding='utf8') as f:
            for x, y in data.items():
                content_len = y['context_len']
                f.write(tokenizer.sequence_to_text(y['context_indices'][:content_len], tokenizer.idx2word) + '\n')
                f.write(tokenizer.sequence_to_text(y['pos_indices'][:content_len], tokenizer.idx2pos) + '\n')
                f.write(tokenizer.sequence_to_text(y['polar_indices'][:content_len], tokenizer.idx2polar) + '\n')
                t = ''
                for i in range(len(y['polarity'])):
                    sidx, eidx = y['aspect_boundary'][i][0], y['aspect_boundary'][i][1]
                    aspect = ' '.join(x.split()[sidx:eidx + 1])
                    polarity = y["polarity"][i]
                    t += aspect + ' ' + str(sidx) + ',' + str(eidx) + ' ' + str(polarity) + '\t'
                t += str(content_len - 1)
                f.write(t + '\n' * 2)

    def statistic(self):
        # 统计数据集
        data = self.data
        tokenizer = self.tokenizer
        sentences = 0
        aspects = set()
        pos, neg, neu = 0, 0, 0
        for x, y, len_x, len_y, *mask in data:
            # x:content,pos,polar y:[as,ae polar,...,eos_position]
            content = tokenizer.sequence_to_text(x[0][:len_x], tokenizer.idx2word).split()
            for i in range(0, len_y - 1, 3):
                aspect_start, aspect_end, polar = y[i:i + 3]
                aspect = ' '.join(content[aspect_start:aspect_end + 1])
                aspects.add(aspect)
                if polar == tokenizer.polar2idx[POS_LABEL]:
                    pos += 1
                elif polar == tokenizer.polar2idx[NEU_LABEL]:
                    neu += 1
                else:
                    neg += 1
            sentences += 1
        print('dataset:[{}] \nsentences:{} aspects:{} [ pos:{} neu:{} neg:{} ]\n'.format(
            self.dataset_name, sentences, len(aspects), pos, neu, neg))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # test()
    tokenizer = build_tokenizer(fnames='data/semeval14', max_seq_len=85)
    ABSADataset(fname='data/semeval14/Laptops_Train.xml.seg', tokenizer=tokenizer)
    ABSADataset(fname='data/semeval14/Laptops_Test_Gold.xml.seg', tokenizer=tokenizer)
    ABSADataset(fname='data/semeval14/Restaurants_Train.xml.seg', tokenizer=tokenizer)
    ABSADataset(fname='data/semeval14/Restaurants_Test_Gold.xml.seg', tokenizer=tokenizer)

    build_embedding_matrix(tokenizer.word2idx)
