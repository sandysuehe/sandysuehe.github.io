---
layout: article
title: "二分搜索Binary Search"
date: 2017-12-27 17:42:41 +0800
categories: algorithm
toc: true
ads: true
image:
    teaser: /teaser/teaser.jpg
---
>简介：本文主要介绍了二分搜索算法的思想和变形应用

### 二分搜索算法的思想

二分搜索法（又名二分查找法）主要是解决在【一堆数中找出指定的数】这类问题。
应用二分搜索法，必须具有以下两个特征：

- 存储在数组中
- 有序排列

在《编程珠玑》一书中的描述：
>在一个包含x的数组内，二分查找通过对范围的跟踪来解决问题。
>开始时，范围就是整个数组。通过将范围中间的元素与x比较并丢弃一半范围，范围被缩小。
>整个过程一直持续，直到在x被发现，或者那个能够包含t的范围已成为空。


### 第一类：需查找和目标值完全相等的数

#### 1. 标准版二分查找，有序无重复数组。

在有序（非降序）数组中查找一个target值，数组中元素没有重复，找到target则返回该元素对应的index，找不到则返回插入位置。
 
Sample：有数组[2, 4, 5, 6, 9]，target = 6。

那么我们可以写出二分查找法的代码如下：


```python
def binary_search_standard(nums, target):
    left = 0
    right = len(nums) - 1
    while(left <= right):                  #POINT 1
        mid = left + (right - left) / 2    #POINT 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1                 #POINT 3
        else:
            right = mid - 1                #POINT 4
    return left
```


注意点：

- POINT 1：如果是left<right, 那么当target等于nums[len(nums)-1]时，会找不到该值。所以应该写成left<=right。
- POINT 2：中间值下标的计算，如果写成(left+right)/2，left+right可能会溢出，从而导致数组访问出错。所以应该写成left+(left+right)/2。
- POINT 3：当nums[mid] < target，左边界应该+1，即为left=mid+1。
- POINT 4：当nums[mid] > target，右边界应该-1，即为right=mid-1。


#### 2. 找出最小index，有序有重复数组。

在有序（非降序）数组中查找一个target值，数组中元素可能有重复，找到target在数组中对应的index的最小值，找不到则返回-1。


那么我们可以写出二分查找法的代码如下：


```python
def binary_search_minindex(nums, target):
    left = 0
    right = len(nums) - 1
    while(left + 1 < right):                  #POINT 1
        mid = left + (right - left) / 2       #POINT 2
        if nums[mid] >= target:               #POINT 3
            right = mid 
        else:                                 #POINT 4
            left = mid + 1
    if nums[left] == target:                  #POINT 5
        return left
    elif nums[right] == target:
        return right
    else:
        return -1
```

注意点：

- POINT 1：因为数组中有可能有重复数字，所以左右相邻的两个数进行对比，循环结束时候，left和right应该相隔1。
- POINT 2：中间值下标的计算，如果写成(left+right)/2，left+right可能会溢出，从而导致数组访问出错。所以应该写成left+(left+right)/2。
- POINT 3：
- 当nums[mid] == target，并不能直接return mid，因为有可能nums[mid-1]或者nums[mid-2]...也等于target，无法保证现在的mid是最小index，所以应该将right=mid，继续循环。
- 当nums[mid] > target，如果有nums[index] == target，那么index一定小于mid，所以right=mid。

- POINT 4：当nums[mid] < target, 右边界应该+1，即为left=mid+1。
- POINT 5：分别检验nums[left]和nums[right]是否等于target。因为两个值都有可能等于target，取决于中间二分时left和right的更新过程。如果循环结束时nums[start]==nums[right]==target，根据题意应返还left，所以我们先验证left。如果两个值都不是target，则target不存在，返回-1。


#### 3. 找出最大index，有序有重复数组。

在有序（非降序）数组中查找一个target值，数组中元素可能有重复，找到target在数组中对应的index的最大值，找不到则返回-1。

那么我们可以写出二分查找法的代码如下：


```python
def binary_search_maxindex(nums, target):
    left = 0
    right = len(nums) - 1
    while(left + 1 < right):
        mid = left + (right - left) / 2
        if nums[mid] <= target:
            left = mid
        else:
            right = mid - 1
    if nums[right] == target:
        return right
    elif nums[left] == target:
        return left
    else:
        return -1
```


### 第二类： 查找第一个不小于目标值的数

这是比较常见的一类，因为我们要查找的目标值不一定会在数组中出现，也有可能是跟目标值相等的数在数组中并不唯一，而是有多个，那么这种情况下nums[mid] == target这条判断语句就没有必要存在。

在有序（非降序）数组中查找一个可能的”最小”index，使得num[index] > target，数组中元素可能有重复，找不到则返回-1。

Sample：有数组[2, 4, 5, 6, 9]，target = 3。 返回4。

那么我们可以写出二分查找法的代码如下：

```python
def binary_search_closest_right(nums, target):
    left = 0
    right = len(nums) - 1
    while(left + 1 < right):
        mid = left + (right - left) / 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid
    if nums[left] > target:
        return left
    elif nums[right] > target:
        return right
    else:
        return -1
```

### 第三类： 查找最后一个小于目标值的数

在有序（非降序）数组中查找一个可能的”最大”index，使得num[index] < target，数组中元素可能有重复，找不到则返回-1。

Sample：有数组[2, 4, 5, 6, 9]，target = 7。 返回6。

那么我们可以写出二分查找法的代码如下：

```python
def binary_search_closest_left(nums, target):
    left = 0
    right = len(nums) - 1
    while(left + 1 < right):
        mid = left + (right - left) / 2
        if nums[mid] >= target:
            right = mid - 1
        else:
            left = mid
    if nums[right] < target:
        return right
    elif nums[left] < target:
        return left
    else:
        return -1
```

##规律

- 1. 当数组中没有重复元素时：while循环判定条件是(start <= end)，每次start更新为mid + 1，end更新为mid – 1。
- 2. 当数组中含有重复元素时或者要找的值是target的相邻数时，判定条件是(start + 1 < end)，当num[mid] == target时，并不返回mid，而是根据情况跟新start或者end。每次start更新为mid，end也更新为mid即可。