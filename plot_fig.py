import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def less_labeled_data():
    xlabel = 'Number of labeled data '
    ylabel = 'F1 score'

    x = [500, 1000, 1500, 2000]
    y_sup = [59.4, 65.52, 69.6, 70.42]  # res
    y_cvt = [64.4, 67.15, 70.37, 71.52]

    fig = plt.figure(figsize=(10, 5), dpi=80)
    ax1 = fig.add_subplot(1, 2, 2)
    ax2 = fig.add_subplot(1, 2, 1)

    x_locator = MultipleLocator(500)
    y_locator = MultipleLocator(2)

    ax1.xaxis.set_major_locator(x_locator)
    ax1.yaxis.set_major_locator(y_locator)
    ax1.set_xlim(450, 2100)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title('Restaurant14')
    ax1.plot(x, y_sup, marker='o', label='PAN')
    ax1.plot(x, y_cvt, marker='o', label='PAN-CVT')
    ax1.legend()
    ax1.grid()

    ax2.xaxis.set_major_locator(x_locator)
    ax2.yaxis.set_major_locator(y_locator)
    ax2.set_xlim(450, 2100)
    ax2.set_title('Laptop14')
    y_sup = [62.54, 65.99, 67.83, 70.15]  # lap
    y_cvt = [65.57, 69.4, 70.21, 71.21]
    ax2.plot(x, y_sup, marker='o', label='PAN')
    ax2.plot(x, y_cvt, marker='o', label='PAN-CVT')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.legend()
    ax2.grid()
    # annotate
    # for px, py in zip(x, y_sup,y_cvt):
    #     plt.annotate(text=str(py), xy=(px, py), xytext=(px, py + 0.1))
    plt.savefig('state/figures/less_labeled data.png')
    plt.show()


def srd():
    xlabel = 'SRD Threshold'
    ylabel = 'Acc'

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))  # 2*2 的图

    x = range(0, 12, 2)
    y0 = [74.4, 72.99, 74.29, 73.77, 76.18, 73.93]  # laptop
    y1 = [81.58, 81.22, 81.07, 81.13, 80.74, 80.63]  # restaurant
    title = ['Laptop14', 'Restaurant14']
    y = [y0, y1]
    for i, ax in enumerate(axes):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(x, y[i], marker='o')
        ax.set_title(title[i])
        ax.grid()
        for px, py in zip(x, y[i]):
            ax.annotate(text=str(py), xy=(px, py), xytext=(px, py))
    plt.savefig('state/figures/srd threshold.png')
    plt.show()


def lcf_senti():
    xlabel = 'SRD Threshold'
    ylabel = 'sentiment score(1e-2)'

    x = range(2, 12, 2)
    y0 = [4.23, 5.82, 7.37, 8.77, 9.48]  # laptop
    y1 = [8.09, 13.42, 16.02, 17.93, 19.85]  # restaurant
    label = ['Laptop14', 'Restaurant14']
    ax = plt.gca()
    for i, y in enumerate([y0, y1]):
        plt.plot(x, y, marker='o', label=label[i])
        for px, py in zip(x, y):
            ax.annotate(text=str(py), xy=(px, py), xytext=(px + 0.1, py + 0.2))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.savefig('state/figures/lcf senti.png')
    plt.show()


def srd_more_laptop():
    xlabel = 'SRD Threshold'
    ylabel = 'Acc'
    x = range(8, 22, 2)
    y = [76.18, 73.93, 74.45, 74.09, 73.46, 75.24, 74.66]
    ax=plt.gca()
    plt.plot(x,y,marker='o')
    for px, py in zip(x, y):
        ax.annotate(text=str(py), xy=(px, py), xytext=(px + 0.1, py ))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Laptop14')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # less_labeled_data()
    # srd()
    # lcf_senti()
    srd_more_laptop()