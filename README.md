# Distributed-system-on-edge-device

此分布式系统是一项可部署在边缘性集群设备上的神经网络训练程序。 目前提供了针对cifar10、NEU64数据集的识别任务(源程序默认使用resnet32/cifar10)。我们在8台jetson nano、8台raspberry pi 4B的集群中实现了1167.3 image/sec 的训练效率。

项目中提供了一种令牌式的动态容错机制，针对于边缘端设备运行过程中不稳定、容易死机等问题进行了有效的处理。

程序兼容于大多数支持linux系统的设备，支持的设备架构：armv6l、armv7l、aarch64。其余设备可通过在Dockers环境下进行部署使用。

## 安装部署

要安装适合设备架构的tensorflow版本，推荐安装1.14:
```
$ pip3 install tensorflow==1.14
```
安装依赖项：psutil：
```
$ pip3 install psutil
```

查询当前需要启动系统的所有设备的ip地址，并添加到main_control.py文件 ring_key={...,device:[],...}中。

## 运行

1. 所有设备提前运行main_cotrol.py进入初始化等待状态，ring_key={...,device:[],...}中第一个设备最后启动
2. 所有设备启动完成后，集群会以此对各个设备进行状态检测
3. 状态检测完成后，设备进入训练状态

----
This project is creature by Jarvis-cai in fudan university, and now manage by hb. 

[hb_link](https://github.com/hb0019/-Distributed-system-on-edge-device)
