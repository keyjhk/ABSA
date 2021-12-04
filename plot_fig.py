import matplotlib.pyplot as plt
# plot
xlabel = 'weight window'
ylabel = 'acc'
x=[0,2,4,6,8]
y=[81.49,81.93,81.96,81.4,81.49]  # res
y1=[76.33,75.44,75.39,75.6,74.61]  # lap
plt.xlabel(xlabel)
plt.ylabel(ylabel)

title='weight window vs acc'
plt.title(title)
plt.plot(x, y, marker='o',label='restaurant')
# plt.plot(x, y1, marker='o',label='laptop')

# annotate
for px, py in zip(x, y):
    plt.annotate(text=str(py), xy=(px, py), xytext=(px, py + 0.1))
# for px, py in zip(x, y1):
#     plt.annotate(text=str(py), xy=(px, py), xytext=(px, py + 0.1))

plt.savefig('state/figures/{}.png'.format(title))
plt.legend()
plt.show()