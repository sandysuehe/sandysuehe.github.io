---
layout: article
title: "Python图表绘制：matplotlib.pyplot入门"
date: 2017-07-21 17:06:41 +0800
categories: python
---
>简介：本文主要介绍了python的绘图库matplotlib，以及pyplot子库的一些常用函数用法。

### matplotlib介绍

matplotlib 是python最著名的绘图库，它提供了一整套和matlab相似的命令API，十分适合交互式地行制图。而且也可以方便地将它作为绘图控件，嵌入GUI应用程序中。 

 Matplotlib则比较强：Matlab的语法、python语言、latex的画图质量（还可以使用内嵌的latex引擎绘制的数学公式）
 
### matplotlib.pyplot介绍
 
matplotlib的pyplot子库提供了和matlab类似的绘图API，方便用户快速绘制2D图表。

matplotlib.pyplot是命令行式函数的集合，每一个函数都对图像作了修改，比如创建图形，在图像上创建画图区域，在画图区域上画线，在线上标注等。


#### 1. 常用pyplot函数
![pyplot常用函数](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/python/pyplot/pyplot常用函数.png?raw=true)

#### 2. 使用plot()函数画图

绘制一个图像：

- y轴的数值序列是[1,2,3,4]
- x轴数值任意
- y轴有注释“y-axis”

```python
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel("y-axis")
plt.show()
```

![pyplot_0](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/python/pyplot/pyplot_0.png?raw=true)


绘制一个图像：

- x轴的数值序列是[1,2,3,4]
- y轴的数值序列是[1,4,9,16]
- 红色圆点
- x轴坐标范围为(0,6),y轴坐标范围为(0.20)


```python
import matplotlib.pyplot as plt

#b代表color为bule，也可以用r代表color为red
#o代表point
plt.plot([1,2,3,4],[1,4,9,16],'bo')

#axis()函数给出了形如[xmin,xmax,ymin,ymax]的列表，指定了坐标轴的范围。
plt.axis([0,6,0,20])

plt.show()
```

![pyplot_1](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/python/pyplot/pyplot_1.png?raw=true)



绘制一个图像，具有3条曲线：

- 第一条曲线：y=x，红色虚线
- 第二条曲线：y=x^2，蓝色solid
- 第三条曲线：y=x^3，绿色tangle


```python
import numpy as np
import matplotlib.pyplot as plt

#生成一组等差数列，从0开始到5，间隔0.2
t=np.arange(0.0,5.0,0.2)

#构建三条曲线
#第一条曲线：y=x，红色虚线
#第二条曲线：y=x^2，蓝色solid
#第三条曲线：y=x^3，绿色tangle
plt.plot(t, t, 'r--',t, t**2, 'bs', t, t**3, 'g^')
plt.show()
```

![pyplot_2](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/python/pyplot/pyplot_2.png?raw=true)


#### 3. 线的属性

可以用两种方式对线的属性进行设置：

##### 1). 使用参数的关键字

```python
import matplotlib.pyplot as plt
x=np.arange(0.0,5.0,0.2)
y=np.arange(0.0,5.0,0.2)

#设置线的宽度为3.0
plt.plot(x, y, linewidth=3.0)
plt.show()
```

![pyplot_3](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/python/pyplot/pyplot_3.png?raw=true)



##### 2). 使用pyplot的setp()命令

```python
import matplotlib.pyplot as plt
t=np.arange(0.0,5.0,0.2)
lines=plt.plot(t, t)

#设置线的颜色为red，宽度为2.0
plt.setp(lines, 'color', 'r', 'linewidth', 2.0)
plt.show()
```
![pyplot_4](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/python/pyplot/pyplot_4.png?raw=true)



#### 4. 多个图像

下面的小例子是产生两个子图像。

```python
import matplotlib.pyplot as plt
import numpy as np

def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

t1=np.arange(0.0, 5.0, 0.1)
t2=np.arange(0.0, 5.0, 0.02)

plt.figure(1)

#指定一个坐标系，显示为2行，每行一图，显示第一幅图像
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
plt.show()

#指定一个坐标系，显示为2行，每行一图，显示第二幅图像
plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()
```

![pyplot_5_0](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/python/pyplot/pyplot_5_0.png?raw=true)

![pyplot_5_1](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/python/pyplot/pyplot_5_1.png?raw=true)


#### 5. 为图像做文本说明

text()命令可以用于在任意位置添加文本，而xlabel(),ylabel(),title()用来在指定位置添加文本。
所有的text()命令返回一个matplotlib.text.Text实例，也可以通过关键字或者setp()函数对文本的属性进行设置。


```python
import matplotlib.pyplot as plt
import numpy as np

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(1000)

n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()
```

![pyplot_6](https://github.com/sandysuehe/sandysuehe.github.io/blob/master/images/python/pyplot/pyplot_6.png?raw=true)