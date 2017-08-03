---
layout: article 
title: "统计学习方法概论(二) 学习笔记"
date: 2017-07-20 12:48:20 +0800
categories: ml
toc: true
ads: true
image:
    teaser: /teaser/statics_teaser.jpeg
---
>简介：本文主要记录了统计学习方法中模型选择方法，模型预测等概念。

### 一、模型选择方法——正则化与交叉验证

#### 1. 模型选择方法：正则化（regularization）

- 由来：模型选择的典型方法，结构风险最小化策略的实现。
- 定义：正则化，是在经验风险上加一个正则化项（regularizer）或罚项（penalty term）。
- 意义：正则化，表示模型复杂度的单调递增函数。
- 模型越复杂，正则化值就越大。

正则化项，可以是模型参数向量的范数：<br/>
![正则化项](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/statics_learn_2/正则化项.png?raw=true)


#### 2. 模型选择方法：交叉验证（cross validation）

如果给定的样本数据充足，可以使用交叉验证的模型选择方法。

![样本切分](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/statics_learn_2/样本切分.png?raw=true)

由于验证集有足够多的数据，使用交叉验证对模型进行选择是有效的。

##### 交叉验证的基本思想

- 重复地使用数据
- 把给定的数据进行切分，将切分的数据集组合为训练集与测试集
- 在此基础上反复地进行训练、测试以及模型选择

##### 交叉验证的几种方法

![交叉验证](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/statics_learn_2/交叉验证.png?raw=true)

### 二、模型的预测能力——泛化能力

#### 什么是泛化能力？

学习方法的泛化能力（generalization ability）, 是指由该方法学习到的模型对未知数据的预测能力。

#### 评价泛化能力的方法

##### 1). 采取最多的办法：通过【测试误差】来评价学习方法的泛化能力。

因为测试数据集是有限的，很有可能由此得到的评价结果是不可靠的。

##### 2). 更可靠的方法：通过【泛化误差】来评价学习方法的泛化能力。

#### 1. 泛化误差（generalization error）

泛化误差，是所学习到的模型的期望风险。

![泛化误差](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/statics_learn_2/泛化误差.png?raw=true)

- 泛化误差，反映了学习方法的泛化能力。
- 如果一种方法学习的模型比另一种方法学习的模型具有更小的泛化误差，那么这种方法更有效。


#### 2. 泛化误差上界

TODO

### 三、监督学习方法——生成模型与判别模型

- 监督学习方法，可以分为两种：1).生成方法，2).判别方法。
- 监督学习所学到的模型，对应的可分为：1).生成模型，2).判别模型。

![生成方法和判别方法](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/statics_learn_2/生成方法和判别方法.png?raw=true)

### 四、分类问题

#### 1. 分类问题的定义

在监督学习中， 当输出变量Y取有限个离散值时，预测问题称为分类问题。

- 分类器（classifier）：监督学习从数据中学习一个分类模型或分类决策函数。
- 类（class）：监督学习中，可能的输出，叫做类。

#### 2. 分类问题的分类

- 二分类：分类的类别为两个
- 多分类：分类的类别为多个

#### 3. 分类问题的两个过程

分类问题包括两个过程：学习 and 分类

##### 1). 学习过程

在学习过程中，根据已知的训练数据集，利于有效的学习方法学习一个分类器。

##### 2). 分类过程

在分类过程中，利用学习的分类器对新的输入实例进行分类。

- （x1,y1),(x2,y2),(x3,y3)...(xn,yn)是训练数据集
-  学习系统由训练数据学习一个分类器P(Y|X)或者Y=f(X)
-  分类系统通过学到的分类器P(Y|X)或者Y=f(X)对于新的输入实例Xn+1进行分类，即预测其输出的类标记Yn+1

![分类问题](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/statics_learn_2/分类问题.png?raw=true)

#### 4. 评价分类器的指标

##### 1). 准确率（accuracy）

#### 5. 评价二分类问题的指标

##### 1). 精确率（precision）

##### 2). 召回率（recall）

##### 3). F-值（精确率和召回率的调和均值）

### 五、标注问题

### 六、回归问题
