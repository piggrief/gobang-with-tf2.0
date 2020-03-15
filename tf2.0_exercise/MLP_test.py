#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # data = pd.read_excel(".\\datasheet\\AGM.xls")
    #
    # x = data["短途阈值（秒）"]
    # y = data["长短途出租车仿真收益差（元每小时）"]

    # 显示数据
    # plt.plot(x, y)
    # plt.show()

    model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3, ), activation='relu'),
                                 # 隐含层 10个 activation为激活函数，选relu
                                 tf.keras.layers.Dense(1)])  # 输出层
    # 顺序模型（一层层顺序构建的网络结构）

    model.summary()  # 隐含层40个参数，因为输入特征三维，w1,w2,w3,b对应一个隐含层神经元

    model.compile(optimizer='adam',
                  loss='mse')

    x = [[1, 2, 3]]
    y = [4]
    model.fit(x, y, epochs=100)

    new_y = model.predict([[2, 3, 4]])

    print(new_y)

    # print(tf.test.is_gpu_available())