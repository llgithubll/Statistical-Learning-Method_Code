"""
load time 22.65605401992798
train time 8.435207605361938
accuracy rate is: 0.8127
"""

from sklearn.linear_model import Perceptron
import numpy as np
import time


def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    print('start to read data')
    # 存放数据及标记的list
    dataArr = []; labelArr = []
    # 打开文件
    fr = open(fileName, 'r')
    # 将文件按行读取
    for line in fr.readlines():
        # 对每一行数据按切割福','进行切割，返回字段列表
        curLine = line.strip().split(',')

        # Mnsit有0-9是个标记，由于是二分类任务，所以将>=5的作为1，<5为-1
        if int(curLine[0]) >= 5:
            labelArr.append(1)
        else:
            labelArr.append(-1)
        #存放标记
        #[int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一哥元素（标记）外将所有元素转换成int类型
        #[int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
        dataArr.append([int(num)/255 for num in curLine[1:]])

    #返回data和label
    return dataArr, labelArr


if __name__ == '__main__':
    start = time.time()
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    testData, testLabel = loadData('../Mnist/mnist_test.csv')
    print('load time', time.time() - start)

    start = time.time()
    clf = Perceptron(max_iter=30)
    clf.fit(np.array(trainData), np.array(trainLabel))
    acc = clf.score(np.array(testData), np.array(testLabel))
    print('train time', time.time() - start)
    print('accuracy rate is:', acc)
