前言
====

力求每行代码都有注释，重要部分注明公式来源。具体会追求下方这样的代码，学习者可以照着公式看程序，让代码有据可查。

![image](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/CodePic.png)

    
如果时间充沛的话，可能会试着给每一章写一篇博客。先放个博客链接吧：[传送门](http://www.pkudodo.com/)。    

##### 注：其中Mnist数据集已转换为csv格式，由于体积为107M超过限制，改为压缩包形式。下载后务必先将Mnist文件内压缩包直接解压。   

       
       
实现
======

### 第一章 统计学习方法概论：
1. 正则化正则的是什么东西？为什么正则化是有效的？

### 第二章 感知机：
博客：[统计学习方法|感知机原理剖析及实现](http://www.pkudodo.com/2018/11/18/1-4/)      
实现：[perceptron/perceptron_dichotomy.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/perceptron/perceptron_dichotomy.py)

* 感知机只能做2分类

1. 感知机为什么不能学习到异或函数？
感知机是线性分类模型，且要求数据线性可分；异或数据线性不可分；
    
### 第三章 K近邻：
博客：[统计学习方法|K近邻原理剖析及实现](http://www.pkudodo.com/2018/11/19/1-2/)      
实现：[KNN/KNN.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/KNN/KNN.py)

```text
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
```
以上是对MNIST数据集进行分类的效果。sample的维度很多（784列），这可能是造成基于树的模型变慢的原因。


* k近邻三要素：距离度量，k值选择，决策规则（如，多数表决）
* k值减小，模型变复杂，容易过拟合，如果邻近实例恰好是噪声，预测就会出错；k值增大，模型就会变简单，即使不相关（距离远）的实例，也会对预测影响。
* kd树算法更适用于训练实例数 >> 空间维数 的k近邻搜索. 空间维度太多，搜索效率会下降。
* k近邻的每个特征都会参与距离计算，不相关的属性会对分类结果有很大影响（最近邻算法对此特别敏感）（可以使用交叉验证，为每个属性挑选权重）


1. k值怎么选择？ **k一般取较小值(3, 4, 5)，使用交叉验证选择最优k值**；
2. k近邻的适用场景？



### 第四章 朴素贝叶斯：
博客：[统计学习方法|朴素贝叶斯原理剖析及实现](http://www.pkudodo.com/2018/11/21/1-3/)      
实现：[NaiveBayes/NaiveBayes.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/NaiveBayes/NaiveBayes.py)    
      
### 第五章 决策树：
博客：[统计学习方法|决策树原理剖析及实现](http://www.pkudodo.com/2018/11/30/1-5/)      
实现：[DecisionTree/DecisionTree.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/DecisionTree/DecisionTree.py)    
      
### 第六章 逻辑斯蒂回归与最大熵模型：       
博客：逻辑斯蒂回归：[统计学习方法|逻辑斯蒂原理剖析及实现](http://www.pkudodo.com/2018/12/03/1-6/)        
博客：最大熵：[统计学习方法|最大熵原理剖析及实现](http://www.pkudodo.com/2018/12/05/1-7/)        

实现：逻辑斯蒂回归：[Logistic_and_maximum_entropy_models/logisticRegression.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/Logistic_and_maximum_entropy_models/logisticRegression.py)    
实现：最大熵：[Logistic_and_maximum_entropy_models/maxEntropy.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/Logistic_and_maximum_entropy_models/maxEntropy.py)       
      
### 第七章 支持向量机：    
博客：[统计学习方法|支持向量机(SVM)原理剖析及实现](http://www.pkudodo.com/2018/12/16/1-8/)      
实现：[SVM/SVM.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/SVM/SVM.py)    
      
### 第八章 提升方法：
实现：[AdaBoost/AdaBoost.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/AdaBoost/AdaBoost.py)    
      
### 第九章 EM算法及其推广：
实现：[EM/EM.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/EM/EM.py)    
      
### 第十章 隐马尔可夫模型：
实现：[HMM/HMM.py](https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/HMM/HMM.py)    

       
       
联系
======
项目未来短期内不再更新，如有疑问欢迎使用issue，也可添加微信或邮件联系。      
此外如果有需要**MSRA**实习内推的同学，欢迎骚扰。             
**Wechat:** lvtengchao（备注“blog-学校/单位-姓名”）      
**Email:** lvtengchao@pku.edu.cn      
      
