"""
以下数据根据testData的前200条数据进行测试得到
test_num 200
k: 3, algorithm: auto, acc: 0.985, time: 38.499552488327026
k: 5, algorithm: auto, acc: 0.985, time: 34.0736927986145
k: 7, algorithm: auto, acc: 0.985, time: 33.51259231567383
k: 9, algorithm: auto, acc: 0.97, time: 33.39327883720398
k: 11, algorithm: auto, acc: 0.975, time: 33.489667654037476
k: 25, algorithm: auto, acc: 0.97, time: 33.61124324798584

k: 5, algorithm: auto, acc: 0.985, time: 34.13753056526184
k: 5, algorithm: ball_tree, acc: 0.985, time: 30.861292600631714
k: 5, algorithm: kd_tree, acc: 0.985, time: 33.43804955482483
k: 5, algorithm: brute, acc: 0.985, time: 3.819897413253784   ？？？

为什么brute方法这么快？因为其他算法build tree耗时吗？把测试用例增多？维度太多了？
test_num 6000
k: 3, algorithm: brute, acc: 0.961, time: 13.932093143463135
k: 5, algorithm: brute, acc: 0.9586666666666667, time: 15.558129072189331
k: 7, algorithm: brute, acc: 0.9595, time: 16.31393051147461
k: 9, algorithm: brute, acc: 0.9551666666666667, time: 15.201151847839355
k: 11, algorithm: brute, acc: 0.9561666666666667, time: 14.665371417999268
k: 25, algorithm: brute, acc: 0.9478333333333333, time: 15.074190378189087

k: 5, algorithm: auto, acc: 0.9586666666666667, time: 423.38795590400696
k: 5, algorithm: ball_tree, acc: 0.9586666666666667, time: 331.3267414569855
k: 5, algorithm: kd_tree, acc: 0.9586666666666667, time: 421.28932213783264
k: 5, algorithm: brute, acc: 0.9586666666666667, time: 14.77004075050354
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time


def loadData(fileName):
    '''
    加载文件
    :param fileName:要加载的文件路径
    :return: 数据集和标签集
    '''
    print('start read file')
    dataArr = []; labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        dataArr.append([int(num) for num in curLine[1:]])
        labelArr.append(int(curLine[0]))
    return dataArr, labelArr


if __name__ == '__main__':
    trainDataArr, trainLabelArr = loadData('../Mnist/mnist_train.csv')
    testDataArr, testLabelArr = loadData('../Mnist/mnist_test.csv')

    test_num = 6000  # k近邻test要便利整个训练集，非常慢，所以只测200条
    print('test_num', test_num)

    # 交叉验证
    # 不同的k值对时间和准确率的影响
    k_values = [3, 5, 7, 9, 11, 25]
    for k in k_values:
        start = time.time()
        neigh = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
        neigh.fit(np.array(trainDataArr), np.array(trainLabelArr))
        acc = neigh.score(np.array(testDataArr[:test_num]), np.array(testLabelArr[:test_num]))
        print(f'k: {k}, algorithm: {neigh.algorithm}, acc: {acc}, time: {time.time()-start}')

    print()

    k = 5
    # 不同的algorithm对时间和准确率的影响
    algs = ['auto', 'ball_tree', 'kd_tree', 'brute']
    for a in algs:
        start = time.time()
        neigh = KNeighborsClassifier(n_neighbors=k, algorithm=a)
        neigh.fit(np.array(trainDataArr), np.array(trainLabelArr))
        acc = neigh.score(np.array(testDataArr[:test_num]), np.array(testLabelArr[:test_num]))
        print(f'k: {k}, algorithm: {a}, acc: {acc}, time: {time.time()-start}')
