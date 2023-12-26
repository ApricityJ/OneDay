import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei"]
# 解决中文乱码等问题
plt.rcParams["axes.unicode_minus"] = False


def plt_kde(x1, x2):
    plt.figure(figsize=(9, 9))
    sns.kdeplot(x1, shade=True, color='r', label='train')
    sns.kdeplot(x2, shade=True, color='b', label='test')
    plt.legend()
    plt.show()


def plt_dist(x1, x2):
    plt.figure(figsize=(9, 9))
    sns.histplot(x1, shade=True, color='r', label='train')
    sns.histplot(x2, shade=True, color='b', label='test')
    plt.legend()
    plt.show()
