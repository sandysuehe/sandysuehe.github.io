---
layout: article
title: "机器学习-逻辑回归"
date: 2017-07-25 16:16:02 +0800
categories: ml
toc: true
ads: true
image:
    teaser: /teaser/logistic_regress.png
---
>简介：本文主要记录了逻辑回归。

### 1. 逻辑回归（logistic regression）

- 逻辑回归是用来做分类任务的。

- 分类任务的目标是找一个函数，把观测值匹配到相关的类和标签上。

学习算法必须用成对的特征向量和对应的标签来估计匹配函数的参数，从而实现更好的分类效果。

在二元分类（binary classification）中，分类算法必须把一个实例配置两个类别。二元分类案例包括，预测患者是否患有某种疾病，音频中是否含有人声，杜克大学男子篮球队在NCAA比赛中第一场的输赢。多元分类中，分类算法需要为每个实例都分类一组标签。

### 2. 逻辑回归处理二元分类
普通的线性回归假设响应变量呈正态分布，也称为高斯分布（Gaussian distribution ）或钟形曲线（bell curve）。正态分布数据是对称的，且均值，中位数和众数（mode）是一样的。很多自然现象都服从正态分布。比如，人类的身高就服从正态分布，姚明那样的高度极少，在99%之外了。

在某些问题里，响应变量不是正态分布的。比如，掷一个硬币获取正反两面的概率分布是伯努力分布（Bernoulli distribution），又称两点分布或者0-1分布。表示一个事件发生的概率是 PP ，不发生的概率是 1−P1−P ，概率在{0,1}之间。线性回归假设解释变量值的变化会引起响应变量值的变化，如果响应变量的值是概率的，这条假设就不满足了。广义线性回归去掉了这条假设，用一个联连函数(link function)来描述解释变量与响应变量的关系。实际上，在第2章，线性回归里面，我们已经用了联连函数。普通线性回归作为广义线性回归的特例使用的是恒等联连函数(identity link function)，将解释变量的通过线性组合的方式来联接服从正态分布的响应变量。如果响应变量不服从正态分布，就要用另外一种联连函数了。

在逻辑回归里，响应变量描述了类似于掷一个硬币结果为正面的概率。如果响应变量等于或超过了指定的临界值，预测结果就是正面，否则预测结果就是反面。响应变量是一个像线性回归中的解释变量构成的函数表示，称为逻辑函数（logistic function）。一个值在{0,1}之间的逻辑函数如下所示：


