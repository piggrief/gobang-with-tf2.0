#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt


def function_api_model(input_data, if_label_one_hot):
    x = tf.keras.layers.Flatten()(input_data)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=input_data, outputs=output)
    if not if_label_one_hot:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',  # 分类标签用数字编码时使用sparse_categorical_crossentropy
                      metrics=['acc']
                      )
    else:
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',  # 分类标签用one-hot编码时使用categorical_crossentropy
                      metrics=['acc']
                      )
    return model


def MISO_test(input1, input2):
    x1 = tf.keras.layers.Flatten()(input1)
    x2 = tf.keras.layers.Flatten()(input2)

    x = tf.keras.layers.concatenate(x1, x2)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input1, input2], outputs=output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # 分类标签用one-hot编码时使用categorical_crossentropy
                  metrics=['acc']
                  )
    return model


def softmax_model(if_label_one_hot):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 扁平层
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))  # dropout层为了抑制过拟合 rate=0.5代表每次随机丢弃掉50%神经元
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))  # dropout层为了抑制过拟合 rate=0.5代表每次随机丢弃掉50%神经元
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # softmax层
    if not if_label_one_hot:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',  # 分类标签用数字编码时使用sparse_categorical_crossentropy
                      metrics=['acc']
                      )
    else:
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',  # 分类标签用one-hot编码时使用categorical_crossentropy
                      metrics=['acc']
                      )
    return model


if __name__ == '__main__':
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

    # Sequential生成的模型
    model = softmax_model(True)
    # 函数式API生成的模型
    model = function_api_model(,True)

    his = model.fit(train_image, train_label_one_hot, epochs=5,
                    validation_data=(test_image, test_label_one_hot))  # 在测试集上验证

    # 绘制训练集和测试集上的loss和acc
    plt.plot(his.epoch, his.history.get('loss'), label='train_loss')
    plt.plot(his.epoch, his.history.get('val_loss'), label='valid_loss')
    plt.legend()
    plt.figure()
    plt.plot(his.epoch, his.history.get('acc'), label='train_acc')
    plt.plot(his.epoch, his.history.get('val_acc'), label='valid_acc')
    plt.legend()

    plt.show()
    print(model.evaluate(test_image, test_label_one_hot))  # 评价模型

    input()

