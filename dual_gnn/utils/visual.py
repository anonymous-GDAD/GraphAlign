import matplotlib.pyplot as plt
import numpy as np


def visualization(acclist1, acclist2, ratio, opt):
    if opt == 'line':
        line(acclist1=acclist1, acclist2=acclist2, ratio=ratio)


def line(acclist1, acclist2, ratio):
    length = len(acclist1)
    x_axis_data = list(range(length))  # x
    y_axis_data1 = acclist1  # y
    y_axis_data2 = acclist2

    plt.plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.5, linewidth=1, label='source_acc')  # 'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    plt.plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.5, linewidth=1, label='target_acc')

    plt.legend()  # 显示上面的label
    plt.xlabel('time')  # x_label
    plt.ylabel('number')  # y_label
    plt.title("{} ratio".format(ratio))
    # plt.ylim(-1,1)#仅设置y轴坐标范围
    plt.show()


# visualization([1,2,3],[4,5,6],0.2,'line')

def visualize_process(source_acc, target_acc, name):
    x = [i for i in range(len(source_acc))]
    plt.plot(x, source_acc)
    plt.plot(x, target_acc)
    plt.legend(['source', 'target'])
    plt.savefig(name + '.jpg')
