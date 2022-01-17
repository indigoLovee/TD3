# TD3
 TD3 in Pytorch
## 仿真环境
gym中LunarLanderContinuous-v2，状态空间和动作空间均为连续形式
## 环境依赖
* gym
* numpy
* matplotlib
* python3.6
* pytorch1.6
## 文件描述
* buffer.py为经验回放脚本；
* network.py为网络脚本，包括演员网络与评价家网络的实现；
* TD3.py为TD3算法实现脚本；
* train.py为训练脚本，创建好output_images文件后，直接运行即可，运行结束后产生的仿真结果存储在创建的output_images文件夹中；
* test.py为测试脚本，主要用于测试训练效果，同样直接运行即可，默认会保存测试的动态图；
* utils.py为工具脚本，主要放置训练或测试时用到的相关函数。
## 仿真结果
仿真结果详见output_images文件。
