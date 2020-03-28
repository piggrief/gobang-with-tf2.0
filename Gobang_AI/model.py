# !/usr/bin/env python
# -*- coding:utf-8 -*-
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import numpy as np


class PolicyValueNet:
    """
    策略价值网络类.

    目前只有cnn.
    """

    def __init__(self, board_height, board_width, history_num, l2_const=1e-4):
        self.board_height = board_height
        self.board_width = board_width
        self.history_num = history_num

        self.l2_const = l2_const  # coef of l2 penalty
        self.regular = keras.regularizers.l2(l2_const)
        self.policy_net = {}
        self.value_net = {}
        self.model = {}

    def cnn_net(self):
        """
        生成cnn结构的策略价值网络.

        :return: 生成cnn模型
        :rtype: keras.model
        """
        input_data = keras.Input(shape=(self.history_num,
                                        self.board_height,
                                        self.board_width))
        # 卷积层
        net = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     data_format="channels_first", activation="relu",
                     input_shape=(self.history_num, self.board_height,
                                  self.board_width),
                     kernel_regularizer=self.regular)(input_data)

        net = Conv2D(64, kernel_size=3, padding='same',
                     data_format="channels_first", activation="relu",
                     kernel_regularizer=self.regular)(net)

        net = Conv2D(128, kernel_size=3, padding='same',
                     data_format="channels_first", activation="relu",
                     kernel_regularizer=self.regular)(net)

        # 策略网络输出
        policy_net = Conv2D(filters=self.history_num, kernel_size=(1, 1),
                            data_format="channels_first", activation="relu",
                            kernel_regularizer=self.regular)(net)

        policy_net = Flatten()(policy_net)
        self.policy_net = Dense(self.board_width * self.board_height,
                                activation="softmax",
                                kernel_regularizer=self.regular)(policy_net)

        # 价值网络输出
        value_net = Conv2D(filters=2, kernel_size=(1, 1),
                           data_format="channels_first", activation="relu",
                           kernel_regularizer=self.regular)(net)
        value_net = Flatten()(value_net)
        value_net = Dense(64, kernel_regularizer=self.regular)(value_net)
        self.value_net = Dense(1, activation="tanh",
                               kernel_regularizer=self.regular)(value_net)

        # 构建最终网络模型
        self.model = keras.Model(input_data, [self.policy_net, self.value_net])

        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer='Adam', loss=losses)

        return self.model

    def get_action_probs(self, chessboard_list):
        """
        根据棋盘列表获得模型的预测结果，即action_probs落子概率列表.

        :param chessboard_list: 棋盘列表
        :type chessboard_list: None*n*h*w的list
        :return: action_probs:落子概率列表;value:当前局面的赢面(价值)
        :rtype: action_probs:h*w的数组；value：None*1*1的数组

        """

        predict_data = np.array(chessboard_list)
        action_probs, value = self.model.predict(predict_data)
        return action_probs, value

    def train_once(self, chessboard_list, action_probs, winner, learning_rate):

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        input_data = np.array(chessboard_list)
        output_policy = np.array(action_probs)
        output_policy = output_policy.reshape((output_policy.shape[0],
                                               output_policy.shape[1] *
                                               output_policy.shape[2]))
        output_value = np.array(winner)
        loss = self.model.evaluate(input_data, [output_policy, output_value],
                                   batch_size=len(input_data), verbose=0)
        action_probs_buff, _ = self.model.predict(input_data)
        entropy = self_entropy(action_probs_buff)
        # 调整学习率
        keras.backend.set_value(self.model.optimizer.lr, learning_rate)
        self.model.fit(input_data, [output_policy, output_value])
        # self.model.fit(input_data, [output_policy, output_value],
        #                batch_size=len(input_data), verbose=0)
        return loss, entropy
