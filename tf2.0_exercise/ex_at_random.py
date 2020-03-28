# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from fashion_mnist_ex.fashion_mnist import fashion_mnist_data_load
from fashion_mnist_ex.fashion_mnist import plot_loss_acc


def ann_dropout_model():
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))  # 输入层
    model.add(keras.layers.Dense(32, activation='relu'))  # 隐藏层1
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32, activation='relu'))  # 隐藏层2
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))  # 输出层

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # 分类标签用数字编码时使用sparse_categorical_crossentropy
                  metrics=['acc']
                  )

    return model


def ann_dropout_regularizer(_lambda):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))  # 输入层
    model.add(keras.layers.Dense(32, activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(_lambda)))  # 隐藏层1
    model.add(keras.layers.Dense(32, activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(_lambda)))  # 隐藏层2
    model.add(keras.layers.Dense(10, activation='softmax'))  # 输出层

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # 分类标签用数字编码时使用sparse_categorical_crossentropy
                  metrics=['acc']
                  )

    return model


def cnn_model_LeNet_5():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        filters=6, kernel_size=3, strides=1, padding='SAME', activation='relu',
        input_shape=(28, 28, 1)))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))  # 高宽各减半的池化层
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))  # 高宽各减半的池化层
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(120, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # 分类标签用数字编码时使用sparse_categorical_crossentropy
                  metrics=['acc']
                  )
    return model
    # 4个3x3大小的卷积核，步长为1，padding方式：SAME
    # layers.Conv2D(4,kernel_size=(3,4),strides=(2,1),padding='SAME')


def resnet_model(input_shape):
    input_data = tf.keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(
        filters=6, kernel_size=3, strides=1, padding='SAME',
        input_shape=(28, 28, 1))(input_data)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)  # 高宽各减半的池化层
    x = keras.layers.ReLU()(x)

    def res_block(x):
        x_pre = x
        x = keras.layers.Conv2D(
            filters=6, kernel_size=3, strides=1, padding='SAME')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Add()([x_pre, x])
        return x

    for _ in range(8):
        x = res_block(x)

    x = keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)  # 高宽各减半的池化层
    x = keras.layers.ReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(120, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_data, outputs=output)

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # 分类标签用数字编码时使用sparse_categorical_crossentropy
                  metrics=['acc']
                  )
    return model


if __name__ == '__main__':
    # 加载 ImageNet 预训练网络模型，并去掉最后一层
    # resnet = keras.applications.ResNet50(weights='imagenet', include_top=False)

    # cnn_model_1()

    [train_image, train_label, test_image, test_label] = \
        fashion_mnist_data_load()

    train_image = np.expand_dims(train_image, 3)
    test_image = np.expand_dims(test_image, 3)

    model = resnet_model((28, 28, 1))

    # tf.keras.utils.plot_model(model, to_file='G:\\Program\\gobang-with-tf2.0\\figures\\model.png', dpi=300)
    model_buff = model.fit(train_image, train_label, epochs=5,
                           validation_data=(test_image, test_label))  # 在测试集上验证

    plot_loss_acc(model_buff.epoch,
                  model_buff.history.get('loss'),
                  model_buff.history.get('acc'),
                  model_buff.history.get('val_loss'),
                  model_buff.history.get('val_acc'))

    print(model.evaluate(test_image, test_label))  # 评价模型

    print("结束！")
