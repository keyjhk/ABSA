import time

from train import Instructor
from config import DEFAULT_OPTION
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from data_utils import *

saved_dir = 'state/predict/'


class Intersection:
    def __init__(self):
        self.sentence_sup = dict()  # predict results by sup
        self.sentence_cvt = dict()  # predict results by cvt
        self.labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.sentences = None  # intersection sentences
        self.intersection = {label: [] for label in self.labels.keys()}  # sentences (context,aspect)

        self.get_intersection()

    def get_intersection(self):
        # predict on dataset by sup/cvt
        predict_sup = instructor.predict(name=model_cpt_sup)  # fname.pkl
        predict_cvt = instructor.predict(name=model_cpt_cvt)

        # load predict results
        sup = pickle.load(open(predict_sup, 'rb'))
        cvt = pickle.load(open(predict_cvt, 'rb'))
        print('sup [true]:{} [false]:{} [acc]:{}'.format(len(sup['true']), len(sup['false']),
                                                         len(sup['true']) / (len(sup['true']) + len(sup['false']))))
        print('cvt [true]:{} [false]:{} [acc]:{}'.format(len(cvt['true']), len(cvt['false']),
                                                         len(cvt['true']) / (len(cvt['true']) + len(cvt['false']))))

        # intersection sentences ,failed in the SUP but success in  CVT

        for x in sup['false']:
            c, a, t, o = x
            self.sentence_sup[(c, a)] = (t, o)
        for x in cvt['true']:
            c, a, t, o = x
            self.sentence_cvt[(c, a)] = (t, o)

        f_inter_sentence = saved_dir + 'inter_sentences.pkl'
        if os.path.exists(f_inter_sentence):
            print('loading inter sentences……')
            sentences = pickle.loads(open(f_inter_sentence, 'rb'))
        else:
            sup_keys = set(list(self.sentence_sup.keys()))
            cvt_keys = set(self.sentence_cvt.keys())
            sentences = sup_keys.intersection(cvt_keys)  # inter

        labels = self.labels
        sentences = sorted(sentences, key=lambda x: self.sentence_cvt[x][0])  # sorted by target label
        self.sentences = sentences
        # write files
        sentence_sup = self.sentence_sup
        sentence_cvt = self.sentence_cvt
        with open(saved_dir + 'inter_sentences.txt', 'w') as f:
            for x in sentences:
                t_label = labels[sentence_sup[x][0]]
                p_label_sup = labels[sentence_sup[x][1]]
                p_label_cvt = labels[sentence_cvt[x][1]]

                self.intersection[sentence_cvt[x][0]].append((*x, *sentence_cvt[x]))  # c,a,t,o

                f.write('[context]:{}\n[aspect]:{}\n'.format(x[0], x[1]))
                f.write('[target]:{} [p_cvt]:{} [p_sup]:{}\n'.format(sentence_sup[x][0], sentence_cvt[x][1],
                                                                     sentence_sup[x][1]))
                f.write('[target]:{} [p_cvt]:{} [p_sup]:{}\n'.format(t_label, p_label_cvt, p_label_sup, ))
                f.write('=' * 30 + '\n')

        return sentences

    def __iter__(self):
        for label in self.labels:
            for item in self.intersection[label]:
                c, a, t, p_cvt = item  # context , aspect
                p_sup = self.sentence_sup[(c, a)][1]
                tlabel, p_label_cvt, p_label_sup = self.labels[t], self.labels[p_cvt], self.labels[p_sup]

                yield c, a, t, p_cvt, p_sup

    def __len__(self):
        return len(self.sentences)


def analyze_sample(sentence, aspect, polarity):
    sample = (sentence, aspect, polarity)
    # NOTE!Keep the prediction order! Sup first,then cvt
    print('sup predict....')
    instructor.predict(model_cpt_sup, sample)  # export max indices
    print('cvt predict....')
    instructor.predict(model_cpt_cvt, sample)


def heatmap(sentence, aspect='', target=''):
    def count_max(array):
        max_counts = []
        for i in range(len(words)):
            count = (array == i).sum()  # count the times as a maximum for each index
            max_counts.append((i, words[i], count))  # i word count
        return max_counts  # List[(i,word,count)]

    words = sentence.split(' ')
    # max indices exported from decoder
    max_pool_sup = pickle.load(open(saved_dir + 'max_pool_sup.pkl', 'rb')).reshape(1, -1)  # 1,hidden_size
    max_pool_cvt = pickle.load(open(saved_dir + 'max_pool_cvt.pkl', 'rb')).reshape(1, -1)
    max_count_sup = count_max(max_pool_sup)  # List[(i,word,count)]
    max_count_cvt = count_max(max_pool_cvt)
    max_times_sup = np.array([x[2] for x in max_count_sup])  # prepare for plot
    max_times_cvt = np.array([x[2] for x in max_count_cvt])

    # plot
    stack = np.stack([max_times_sup, max_times_cvt])
    fig = plt.figure()
    ax = plt.gca()
    # matshow 返回对象  AxesImage
    cax = ax.matshow(stack, cmap='Blues')
    title = '【{}】 【{}】 【{}】'.format(sentence, aspect, target)
    # ax.set_title(title)

    # 刻度
    # 如果不添加刻度的话 就会默认显示012 这样的数字索引
    # ticklabels 用于显示文字 添加一开始的空字符串
    # ''空字符 会在原点显示 其余字符会在每个矩阵的中间位置显示
    # 也就是说 如果矩阵有n列 除了n个 文字刻度 显示之外 还要在它的头前面加一个 '' 总计n+1
    # y轴同理
    xticklabels = [''] + words
    yticklabels = [''] + ['no-cvt', 'cvt']

    ax.set_xticklabels(xticklabels, rotation=90)  # 设置x轴刻度标签
    ax.set_yticklabels(yticklabels)

    # majorlocator 是用于调整刻度间距的
    # MultipleLocator 表示固定间隔1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(saved_dir + 'figures/' + '{}.png'.format(title))
    plt.show()


if __name__ == '__main__':
    # initial
    model_cpt_sup = 'restaurant_1641178246_cvt_epoch5_acc_75.59_f1_60.04.pkl'
    model_cpt_cvt = 'restaurant_1641178027_cvt_epoch19_acc_77.62_f1_67.03.pkl'
    opt = DEFAULT_OPTION.set({'dataset': 'restaurant'})  # select dataset
    instructor = Instructor(opt)

    # analyze
    intersection = Intersection()
    print(len(intersection))
    for item in intersection:
        c, a, t, p_cvt, p_sup = item
        # print('{}\n{}\n t:[{}] cvt:[{}] sup:[{}]'.format(c, a, t, p_cvt, p_sup))
        analyze_sample(c, a, t - 1)
        heatmap(c, a, intersection.labels[t])
        print('=' * 30)
        time.sleep(1)
        exit()

    # an sample
    # sentence = "this particular location certainly uses substandard meats ."
    # aspect = "meats"
    # polarity = -1  # -1 ,neg:-1 neu:0 pos:2
    # analyze_sample(sentence, aspect, polarity)
    # # heatmap
    # heatmap(sentence)
