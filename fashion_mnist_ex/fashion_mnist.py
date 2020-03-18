#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt


def fashion_mnist_data_load():
    fm_data_sets = tf.keras.datasets.fashion_mnist.load_data()
    train_image = fm_data_sets[0][0]
    train_label = fm_data_sets[0][1]
    test_image = fm_data_sets[1][0]
    test_label = fm_data_sets[1][1]

    # 归一化
    train_image = train_image / 255
    test_image = test_image / 255

    # 转换成one-hot编码
    train_label_one_hot = tf.keras.utils.to_categorical(train_label)
    test_label_one_hot = tf.keras.utils.to_categorical(test_label)

    return [train_image, train_label_one_hot, test_image, test_label_one_hot]


def plot_loss_acc(epoch, loss=[], acc=[], val_loss=[], acc_val=[]):
    # 绘制训练集和测试集上的loss和acc
    if [] != loss or [] != val_loss:
        plt.figure()
    if [] != loss:
        plt.plot(epoch, loss, label='train_loss')
    if [] != val_loss:
        plt.plot(epoch, val_loss, label='valid_loss')
    if [] != loss or [] != val_loss:
        plt.legend()

    if [] != acc or [] != acc_val:
        plt.figure()
    if [] != acc:
        plt.plot(epoch, acc, label='train_acc')
    if [] != acc_val:
        plt.plot(epoch, acc_val, label='valid_acc')
    if [] != acc or [] != acc_val:
        plt.legend()

    plt.show()
