---
layout: article
title: "机器学习-线性回归"
date: 2017-07-24 14:45:26 +0800
categories: ml
---
> 线性回归概要
![线性回归概要](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/linear_regression/线性回归概要.png?raw=true)

## 一、回归问题简介

### 1. 回归问题

预测一个与对象相关的值连续的属性。

### 2. 应用场景

药品反应，股票价格。

### 3. 模型

目标值y是输入变量x的线性组合。
用数学表达：预测值为
![线性回归问题](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/linear_regression/线性回归问题.png?raw=true)

这个模块中，我们定义

- 向量w=(w1,w2,...wp)为coef_
- w0为intercept_


## 二、线性回归(Linear Regression)

### 1. 普通最小二乘法
线性回归(Linear Regression)用系数(w1,w2,...wp)来拟合一个线性模型，使得数据集实际观测数据（实际值）和预测数据（估计值）之间存在的差平方和最小。
![普通最小二乘法](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/linear_regression/普通最小二乘法.png?raw=true)

线性回归(Linear Regression)模型会调用fit方法来拟合X，y（X为输入，y为输出）。并且把拟合的线性模型的系统w存储到成员变量coef_中。


### 2. 一元线性回归

#### 1). 一元线性回归模型
    y=α+βx
   
#### 2). 一元线性回归拟合模型的参数估计常用方法
- 普通最小二乘法（ordinary least squares）
- 线性最小二乘法（linear least squares）


```python
import matplotlib.pyplot as plt

plt.figure()
plt.title(u'Price vs D')
plt.xlabel(u'D')
plt.ylabel(u'Price')
plt.axis([0, 25, 0, 25])
plt.grid(True)

from sklearn import linear_model

X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

clf=linear_model.LinearRegression()
clf.fit(X,y)

print 'w1，w2,...wp', clf.coef_
print 'w0', clf.intercept_

X2=[[0], [10], [14], [25]]
y2=clf.predict(X2)

#实际值的坐标点
plt.plot(X, y, 'k.')
#预估值的模型曲线
plt.plot(X2,y2,'g-')
plt.show()
```

    w1，w2,...wp [[ 0.9762931]]
    w0 [ 1.96551724]

![最小二乘法曲线](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/linear_regression/最小二乘法曲线.png?raw=true)


#### 3). 带成本函数的模型拟合评估

- 成本函数（cost function）

  用来定义模型与观测值的误差。也叫损失函数（loss function）。

- 训练误差（training errors）

  模型预测的价格与训练集数据的差异称为残差（residuals）或训练误差。
  
- 测试误差（test errors）
  模型计算测试集，那时模型预测的价格与测试集数据的差异称为预测误差（prediction errors）或测试误差（test errors）。

 通过残差之和最小化实现最佳拟合，也就是说模型预测的值与训练集的数据最接近就是最佳拟合。对模型的拟合度进行评估的函数称为残差平方和（residual sum of squares）成本函数。就是让所有训练数据与模型的残差的平方之和最小化。
 
```python
import numpy as np
print('残差平方和: %.2f' % np.mean((model.predict(X) - y) ** 2))
```

残差平方和: 1.75
 
模型的训练误差，是训练样本点与线性回归模型的纵向距离，如下图所示。


```python
import matplotlib.pyplot as plt

plt.figure()
plt.title(u'Price vs D')
plt.xlabel(u'D')
plt.ylabel(u'Price')
plt.axis([0, 25, 0, 25])
plt.grid(True)

from sklearn import linear_model

X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

clf=linear_model.LinearRegression()
clf.fit(X, y)

x2=[[0], [10], [14], [25]]
y2=clf.predict(x2)

#绘制实际坐标点
plt.plot(X, y, 'k.')
#绘制预估模型曲线
plt.plot(x2, y2, 'g-')

#绘制残差预测集
yr=model.predict(X)
for idx, x in enumerate(X):
    plt.plot([x, x], [y[idx], yr[idx]], 'r-')
plt.show()
```


![成本函数](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/linear_regression/成本函数.png?raw=true)

####4). 模型评估

如何评价模型在现实中的表现呢？
现在让我们假设有另一组数据，作为测试集进行评估。<br/>
我们使用R方（r-squared）评估模型预测的效果。R方也叫确定系数（coefficient of determination），表示模型对现实数据拟合的程度。

![R方](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/linear_regression/R方.png?raw=true)

scikit-learn样例

```python
X_test = [[8], [9], [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]]
clf = linear_model.LinearRegression()
clf.fit(X, y)
clf.score(X_test, y_test)
```

结果：
0.6620052929422553
    


### 3. 多元线性回归 

#### 1). 多元线性回归模型
    y=α+β1x1+β2x2+⋯+βnxn
    
![多元线性回归](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/linear_regression/多元线性回归.png?raw=true)


#### 2). scikit-learn样例

```python
import matplotlib.pyplot as plt

plt.figure()
plt.title(u'Price vs D')
plt.xlabel(u'D')
plt.ylabel(u'Price')
plt.axis([0, 25, 0, 25])
plt.grid(True)

from sklearn import linear_model

X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]

clf=linear_model.LinearRegression()
clf.fit(X,y)

X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]
predictions=clf.predict(X_test)

#绘制实际坐标点
plt.plot(X, y, 'k.')
#绘制预估模型曲线
plt.plot(X_test, y_test, 'g-')
plt.show()

for i, prediction in enumerate(predictions):
     print('Predicted: %s, Target: %s' % (prediction, y_test[i]))
        
print('R-squared: %.2f' % clf.score(X_test, y_test))


```

![多元线性回归例子](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/linear_regression/多元线性回归例子.png?raw=true)

    Predicted: [ 10.0625], Target: [11]
    Predicted: [ 10.28125], Target: [8.5]
    Predicted: [ 13.09375], Target: [15]
    Predicted: [ 18.14583333], Target: [18]
    Predicted: [ 13.3125], Target: [11]
    R-squared: 0.77



增加解释变量让模型拟合效果更好了。多元回归确实比一元回归效果更好。


### 4. 多项式回归 

假如解释变量和响应变量的关系不是线性的呢？下面我们来研究一个特别的多元线性回归的情况，可以用来构建非线性关系模型。

#### 1). 多项式回归模型
![二次回归](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/linear_regression/二次回归.png?raw=true)

#### 2). scikit-learn样例

```python
import matplotlib.pyplot as plt

plt.figure()
plt.title('Price vs diameter')
plt.xlabel('diameter')
plt.ylabel('price')
plt.axis([0, 25, 0, 25])
plt.grid(True)
    
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

#线性回归
regressor=linear_model.LinearRegression()
regressor.fit(X_train, y_train)

#获取0到26之间的100个数，间距相等
xx=np.linspace(0,26,100)

#将列表中的每个元素转化为列表，再做预测
yy=regressor.predict(xx.reshape(xx.shape[0],1))

#训练集的多项式化（二次回归）
quadratic_featurizer = preprocessing.PolynomialFeatures(degree=2)
X_train_quadratic=quadratic_featurizer.fit_transform(X_train)
X_test_quadratic=quadratic_featurizer.transform(X_test)

#多项式线性回归（二次回归）
regressor_quadratic=linear_model.LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)

#预测参数的多项式化（二次回归）
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

#训练集的多项式化（七次回归）
seventh_featurizer = preprocessing.PolynomialFeatures(degree=7)
X_train_seventh = seventh_featurizer.fit_transform(X_train)
X_test_seventh = seventh_featurizer.transform(X_test)

#多项式线性回归（七次回归）
regressor_seventh = linear_model.LinearRegression()
regressor_seventh.fit(X_train_seventh, y_train)

#预测参数的多项式化（七次回归）
xx_seventh = seventh_featurizer.transform(xx.reshape(xx.shape[0], 1))

plt.plot(X_train, y_train, 'k.')
plt.plot(xx, yy)
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')
plt.plot(xx, regressor_seventh.predict(xx_seventh))

plt.show()

#R方值
print('liner_model r-squared', regressor.score(X_test, y_test))
print('second liner r-squared', regressor_quadratic.score(X_test_quadratic, y_test))
print('seventh liner r-squared', regressor_seventh.score(X_test_seventh, y_test))
```

![多项式回归例子](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/ml/linear_regression/多项式回归例子.png?raw=true)

    ('liner_model r-squared', 0.80972679770766498)
    ('second liner r-squared', 0.86754436563451076)
    ('seventh liner r-squared', 0.49198460568655122)

效果如上图所示，直线为一元线性回归（R方0.81），曲线为二次回归（R方0.87），其拟合效果更佳。还有三次回归，就是再增加一个立方项（ β3x3β3x3 ）。同样方法拟合，七次拟合的R方值更低，虽然其图形基本经过了所有的点。可以认为这是拟合过度（over-fitting）的情况。这种模型并没有从输入和输出中推导出一般的规律，而是记忆训练集的结果，这样在测试集的测试效果就不好了。


### 4. 梯度下降法（gradient descent）

梯度下降法被比喻成一种方法，一个人蒙着眼睛去找从山坡到溪谷最深处的路。他看不到地形图，所以只能沿着最陡峭的方向一步一步往前走。每一步的大小与地势陡峭的程度成正比。如果地势很陡峭，他就走一大步，因为他相信他仍在高出，还没有错过溪谷的最低点。如果地势比较平坦，他就走一小步。这时如果再走大步，可能会与最低点失之交臂。如果真那样，他就需要改变方向，重新朝着溪谷的最低点前进。他就这样一步一步的走啊走，直到有一个点走不动了，因为路是平的了，于是他卸下眼罩，已经到了谷底深处，小龙女在等他。

梯度下降法会在每一步走完后，计算对应位置的导数，然后沿着梯度（变化最快的方向）相反的方向前进。总是垂直于等高线。

需要注意的是，梯度下降法来找出成本函数的局部最小值。一个三维凸（convex）函数所有点构成的图行像一个碗。碗底就是唯一局部最小值。非凸函数可能有若干个局部最小值，也就是说整个图形看着像是有多个波峰和波谷。梯度下降法只能保证找到的是局部最小值，并非全局最小值。残差平方和构成的成本函数是凸函数，所以梯度下降法可以找到全局最小值。

梯度下降法的一个重要超参数是步长（learning rate），用来控制蒙眼人步子的大小，就是下降幅度。如果步长足够小，那么成本函数每次迭代都会缩小，直到梯度下降法找到了最优参数为止。但是，步长缩小的过程中，计算的时间就会不断增加。如果步长太大，这个人可能会重复越过谷底，也就是梯度下降法可能在最优值附近摇摆不定。

如果按照每次迭代后用于更新模型参数的训练样本数量划分，有两种梯度下降法。批量梯度下降（Batch gradient descent）每次迭代都用所有训练样本。随机梯度下降（Stochastic gradient descent，SGD）每次迭代都用一个训练样本，这个训练样本是随机选择的。当训练样本较多的时候，随机梯度下降法比批量梯度下降法更快找到最优参数。批量梯度下降法一个训练集只能产生一个结果。而SGD每次运行都会产生不同的结果。SGD也可能找不到最小值，因为升级权重的时候只用一个训练样本。它的近似值通常足够接近最小值，尤其是处理残差平方和这类凸函数的时候。

下面我们用scikit-learn的SGDRegressor类来计算模型参数。它可以通过优化不同的成本函数来拟合线性模型，默认成本函数为残差平方和。


```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

#使用sklearn自带数据集，分割训练集和测试集。
data=load_boston()
X_train, X_test, y_train, y_test=train_test_split(data.data, data.target)

#StandardScaler做归一化处理
X_scaler=StandardScaler()
y_scaler=StandardScaler()

#训练集和测试集做归一化处理
X_train=X_scaler.fit_transform(X_train)
y_train=y_scaler.fit_transform(y_train)
X_test=X_scaler.transform(X_test)
y_test=y_scaler.transform(y_test)

#用交叉验证方法训练模型和测试模型
regressor=SGDRegressor(loss='squared_loss')
scores=cross_val_score(regressor, X_train, y_train, cv=5)
print 'cross_validation R-squared value: ', scores
print 'cross_validation R-squared mean value: ', np.mean(scores)

regressor.fit_transform(X_train, y_train)
print 'test R-squared value: ', regressor.score(X_test, y_test)
```

    cross_validation R-squared value:  [ 0.67766165  0.53783947  0.83870029  0.64796039  0.68963189]
    cross_validation R-squared mean value:  0.678358738602
    test R-squared value:  0.724030526759
