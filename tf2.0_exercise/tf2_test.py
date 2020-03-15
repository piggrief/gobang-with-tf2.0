#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = pd.read_excel(".\\datasheet\\AGM.xls")

    x = data["短途阈值（秒）"]
    y = data["长短途出租车仿真收益差（元每小时）"]

    # 显示数据
    # plt.plot(x, y)
    # plt.show()

    model = tf.keras.Sequential()  # 顺序模型（一层层顺序构建的网络结构）

    model.add(tf.keras.layers.Dense(1, input_shape=(1, )))  # 添加Dense层(ax+b)

    model.summary()  # 显示模型结构

    model.compile(optimizer='adam',  # 优化方法 'adam'：梯度下降
                  loss='mse',  # 损失函数 'mse'：均方误差
                  )
    history = model.fit(x, y, epochs=200)  # 模型训练

    new_y = model.predict(x)  # 模型预测

    plt.plot(x, y, 'r')
    plt.plot(x, new_y, 'b')
    plt.show()

    # print(tf.test.is_gpu_available())
