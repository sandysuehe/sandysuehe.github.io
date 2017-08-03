---
layout: article 
title: "机器学习-理解ROC和AUC"
date: 2017-07-12 11:38:25 +0800
categories: ml
---
> 简介：本文主要介绍了机器学习中ROC和AUC的概念。
 

### 一、二分类模型的理想状态是什么呢？

- 理想的二分类模型能将原本是对的预测为对，原本是错的预测为错。
- 一般情况下，我们很难收集到完备的“原本是对的，原本是错的”数据集，也就是说，通常情况下我们获得是完备的“原本是对的，原本是错的”数据集的一个子集。
- 因此，评价二分类模型的优劣就是在该子集上进行的。我们希望在该子集上对二分类模型的评价是无偏的，也就是说，在概率上保证在该子集上对二分类模型的评价与在完备集上的评价一致。


### 二、如何评价两个二分类模型的好坏呢？

二分类问题的预测结果可能正确，也可能不正确。

- 结果正确存在两种可能：原本对的预测为对 TP（即True Positives），原本错的预测为错 FN（即False Negatives）；
- 结果错误也存在两种可能：原本对的预测为错 TN（即True Neagtives），原本错的预测为对 FP（即False Positives）；


#### 1. 评价二分类模型好坏的四个基本元素

        1). 原本是对的预测为对的个数 TP；
        2). 原本是错的预测为错的个数 FN;
        3). 原本是对的预测为错的个数 TN;
        4). 原本是错的预测为对的个数 FP。
        
#### 2. 评价二分类模型好坏的评价指标

评价一个模型的好坏用四个参数是不是有点不太直观，要是只有一个评价指标。
如果一个模型的这指标比别的模型大，那这个模型就比别的模型好。（或者反过来，一个模型的这指标比别的模型小，那这个模型比别的模型好）
这个指标就是下面要提到的AUC。


### 三、AUC概念

- ROC（Receiver Operating Characteristic）曲线和AUC（Area Under Curve）常被用来评价一个二值分类器（binary classifier）的优劣。
- 相比准确率、召回率、F-score这样的评价指标，ROC曲线有这样一个很好的特性：当测试集中正负样本的分布变化的时候，ROC曲线能够保持不变。在实际的数据集中经常会出现类不平衡（class imbalance）现象，即负样本比正样本多很多（或者相反），而且测试数据中的正负样本的分布也可能随着时间变化。

![正负样本分布](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/roc_auc/ml.jpeg?raw=true)

  如Fig 1左侧所示。
  
  - 其中Positives代表预测是对的，Negatives代表预测是错的；True代表原本为对，False代表原本为错。
  - 原本对的预测为对 TP，原本错的预测为错 FN， 原本对的预测为错TN，原本错的预测为对FP。
  - P = TP + FN
  - N = TN + FP

#### 1). FPR(fp rate or false positive rate)
     原本是错的预测为对的比例
     fp = FP／N
     越小越好，0为理想状态
     
#### 2). TPR(tp rate or true positive rate)
     原本是对的预测为对的比例
     tp rate = TP/P
     越大越好，1为理想状态
     
     
#### 3). ROC曲线(Receiver Operating Characteristic)
    得到算法的一组（fp rate， tp rate）然后做出的曲线（没办法用大小来衡量）

![ROC曲线](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/roc_auc/ROC.png?raw=true)
  
 正如上面的ROC曲线的示例图所示，ROC曲线的横坐标为false positive rate（FPR），纵坐标为true positive rate（TPR）。
 
 接下来考虑ROC曲线图中的四个点和一条线。
 
 - 第一个点，(0,1)，即FPR=0, TPR=1，这意味着FN（false negative）=0，并且FP（false positive）=0。说明这是一个完美的分类器，它将所有的样本都正确分类。
 - 第二个点，(1,0)，即FPR=1，TPR=0，说明这是一个最糟糕的分类器，因为它成功避开了所有的正确答案。
 - 第三个点，(0,0)，即FPR=TPR=0，即FP（false positive）=TP（true positive）=0，可以发现该分类器预测所有的样本都为负样本（negative）。
 - 第四个点（1,1），分类器实际上预测所有的样本都为正样本。

 经过以上的分析，我们可以断言，ROC曲线越接近左上角，该分类器的性能越好。
 
 - 下面考虑ROC曲线图中的虚线y=x上的点。这条对角线上的点其实表示的是一个采用随机猜测策略的分类器的结果，例如(0.5,0.5)，表示该分类器随机对于一半的样本猜测其为正样本，另外一半的样本为负样本。


##### 如何绘制ROC曲线
对于一个特定的分类器和测试数据集，显然只能得到一个分类结果，即一组FPR和TPR结果，而要得到一个曲线，我们实际上需要一系列FPR和TPR的值。

  假如我们已经得到了所有样本的概率输出（属于正样本的概率），现在的问题是如何改变“discrimination threashold”？我们根据每个测试样本属于正样本的概率值从大到小排序。下图是一个示例，图中共有20个测试样本，“Class”一栏表示每个测试样本真正的标签（p表示正样本，n表示负样本），“Score”表示每个测试样本属于正样本的概率。
  
  ![样本分布](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/roc_auc/sample.png?raw=true)
  
  我们从高到低，依次将“Score”值作为阈值threshold，当测试样本属于正样本的概率大于或等于这个threshold时，我们认为它为正样本，否则为负样本。举例来说，对于图中的第4个样本，其“Score”值为0.6，那么样本1，2，3，4都被认为是正样本，因为它们的“Score”值都大于等于0.6，而其他样本则都认为是负样本。每次选取一个不同的threshold，我们就可以得到一组FPR和TPR，即ROC曲线上的一点。这样一来，我们一共得到了20组FPR和TPR的值，将它们画在ROC曲线的结果如下图：
  
  ![ROC绘图](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/roc_auc/roc_pic.png?raw=true)
  
-  当我们将threshold设置为1和0时，分别可以得到ROC曲线上的(0,0)和(1,1)两个点。将这些(FPR,TPR)对连接起来，就得到了ROC曲线。当threshold取值越多，ROC曲线越平滑。

- 其实，我们并不一定要得到每个测试样本是正样本的概率值，只要得到这个分类器对该测试样本的“评分值”即可（评分值并不一定在(0,1)区间）。评分越高，表示分类器越肯定地认为这个测试样本是正样本，而且同时使用各个评分值作为threshold。我认为将评分值转化为概率更易于理解一些。

##### 为什么使用ROC曲线
- 因为ROC曲线有个很好的特性：当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变。在
- 实际的数据集中经常会出现类不平衡（class imbalance）现象，即负样本比正样本多很多（或者相反），而且测试数据中的正负样本的分布也可能随着时间变化。下图是ROC曲线和Precision-Recall曲线的对比：

  ![ROC与PR曲线对比](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/roc_auc/compare.png?raw=true)
  
 在上图中，(a)和(c)为ROC曲线，(b)和(d)为Precision-Recall曲线。(a)和(b)展示的是分类其在原始测试集（正负样本分布平衡）的结果，(c)和(d)是将测试集中负样本的数量增加到原来的10倍后，分类器的结果。可以明显的看出，ROC曲线基本保持原貌，而Precision-Recall曲线则变化较大。
    
#### 4). AUC(Area Under Curve)
    由来：因为ROC曲线没办法用大小来衡量，所以引入AUC的概念。
    含义：ROC曲线下的面积
    越大越好，1为理想状态
    
    
##### AUC的含义
    The AUC value is equivalent to the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.
- AUC值是一个概率值，当你随机挑选一个正样本以及一个负样本，当前的分类算法根据计算得到的Score值将这个正样本排在负样本前面的概率就是AUC值。
- 当然，AUC值越大，当前的分类算法越有可能将正样本排在负样本前面，即能够更好的分类。

##### AUC的计算
- AUC（Area Under Curve）被定义为ROC曲线下的面积，这个面积的数值不会大于1。
- 由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围在0.5和1之间。使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好。


