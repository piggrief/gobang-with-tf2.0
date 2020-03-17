#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = tf.random.normal((2, 4, 4, 3), 5, 5)
    print(data.shape)
    new_data = tf.reshape(data, [2, 3, 16])
    result_data = tf.reshape(new_data, [2, 4, 4, 3])

    print("结束！")
