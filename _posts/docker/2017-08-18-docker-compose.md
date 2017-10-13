---
layout: article
title: "Docker Compose介绍及使用"
date: 2017-08-18 15:06:41 +0800
categories: docker 
toc: true
ads: true
image:
    teaser: /teaser/teaser.jpg
---
>简介：Docker Compose负责快速在集群中部署分布式应用。

### Docker Compose介绍

Docker Compose是定义和运行多个Docker容器的应用(Defining and running multi-container Docker application)

在日常工作中，经常会碰到需要多个容器相互配合来完成某项任务的情况。例如要实现一个 Web 项目，除了 Web 服务容器本身，往往还需要再加上后端的数据库服务容器，甚至还包括负载均衡容器等。

Docker Compose允许用户通过一个单独的 docker-compose.yml 模板文件（YAML 格式）来定义一组相关联的应用容器为一个项目（project）。

### Docker Compose的两个重要概念 
 
#### 1. 服务Service 

一个应用的容器，实际上可以包括若干运行相同镜像的容器实例。

#### 2. 项目Project 

由一组关联的应用容器组成的一个完整业务单元，在 docker-compose.yml 文件中定义。


Compose 的默认管理对象是项目，通过子命令对项目中的一组容器进行便捷地生命周期管理。

Compose 项目由 Python 编写，实现上调用了 Docker 服务提供的 API 来对容器进行管理。因此，只要所操作的平台支持 Docker API，就可以在其上利用 Compose 来进行编排管理。

### 未完待续...

