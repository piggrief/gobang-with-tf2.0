#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import math
import copy
import time
from Gobang_AI.gobang_game import print_chessboard
from Gobang_AI.model import PolicyValueNet


class MCTSTreeNode:
    """
    MCTS树节点类
    """
    def __init__(self, parent, node_action, node_prob):
        self.parent = parent  # 父节点
        self.children = []  # 子节点
        self._n = 1  # 访问次数
        self._Q = 0  # Q值
        self._P = node_prob  # P值(当前局面该落子的概率)
        self.action = node_action

        self.score = 0

    def expand(self, action_prob):
        """
        拓展一个未展开节点
        :param action_prob:策略价值网络输出的棋盘二维局面每个点的落子概率 是h*w的数组
        :return: 无
        """
        for i in range(len(action_prob)):
            for j in range(len(action_prob[i])):
                self.children.append(MCTSTreeNode(self, [i, j], action_prob[i][j]))

    def update_info(self, leaf_value):
        """
        用叶节点胜负的值更新父节点的n和Q
        :param leaf_value:叶节点值
        :return:无
        """
        self._n += 1
        self._Q += (leaf_value - self._Q) / self._n

    def back_propagation(self, leaf_value):
        """
        由胜负节点反向传播至根节点
        :param leaf_value: 胜负节点（叶节点）的值
        :return: 无
        """
        if self.parent:
            self.parent.back_propagation(-leaf_value)
        self.update_info(leaf_value)

    def select(self, _c):
        """
        选择Q+c*p*sqrt(parent.n)/(1+n)最大的子节点
        :param _c:探索率，制约u(P)和Q之间的关系
        :return: Q+u（P）最大的子节点
        """
        max_score = -math.inf
        best_child = self.children[0]
        # 找最大的Q+u(P)
        for child in self.children:
            if child._P < 0:
                u = -math.inf
            else:
                u = _c * child._P * np.sqrt(child.parent._n) / (1 + child._n)
            score = child._Q + u
            child.score = score
            if max_score < score:
                max_score = score
                best_child = child
        return best_child

    def if_leaf_node(self):
        return [] == self.children

    def if_root_node(self):
        return self.parent is None


class MCTS:
    def __init__(self, chessboard_height=5, chessboard_width=5,
                 chessboard_history_num=4, simulation_num=300,
                 _c=5, if_print_sim_once=False):
        self.chessboard_height = chessboard_height
        self.chessboard_width = chessboard_width
        self.chessboard_history_num = chessboard_history_num
        self.root_node = MCTSTreeNode(None, (-1, -1), 1)
        self.init_chessboard = np.zeros(shape=(chessboard_height, chessboard_width), dtype=float)  # 0为没有，1为P1，2为P2
        self.init_chessboard = self.init_chessboard.tolist()

        self._c = _c
        self.simulation_num = simulation_num

        self.if_print_sim_once = if_print_sim_once

        self.vp_net = PolicyValueNet(chessboard_height, chessboard_width, chessboard_history_num)
        self.vp_net.model = self.vp_net.cnn_net()

    def renew_chessboard_list(self, chessboard_list, chessboard):
        """
        用这次记录的棋盘动态更新棋盘列表
        :param chessboard_list: 待更新的棋盘列表
        :param chessboard: 这次棋盘
        :return:棋盘列表 (内部已经更新self.chessboard_list) n*h*w
        """
        chessboard_list.append(copy.deepcopy(chessboard))
        diff_value = self.chessboard_history_num - len(chessboard_list)
        if diff_value > 0:
            # 填充空历史棋盘
            for _ in range(diff_value):
                empty_list = np.zeros(shape=(self.chessboard_height,
                                             self.chessboard_width),
                                      dtype=float).tolist()
                chessboard_list.insert(0, empty_list)
        elif diff_value < 0:
            chessboard_list = chessboard_list[-diff_value:]
        return chessboard_list

    @staticmethod
    def do_action(chessboard, action, player):
        """
        执行玩家player落子action，更新self.chessboard
        :param chessboard: 执行落子的棋盘
        :param action: 落子
        :param player: 玩家
        :return: self.chessboard
        """
        chessboard[action[0]][action[1]] = player
        return chessboard

    @staticmethod
    def reverse_player(player):
        if 1 == player:
            return 2
        else:
            return 1

    def get_action_prob(self, chessboard_list):
        """
        输入棋盘列表（n个历史棋盘）并通过价值策略网络生成下一步落子概率数组（h*w）
        :param chessboard_list: 棋盘列表（n个历史棋盘） n*h*w
        :return: 下一步落子概率数组（h*w）
        """
        # 策略价值网络预测
        input_data = np.array(chessboard_list)
        input_data = np.expand_dims(input_data, axis=0)
        action_prob, value = self.vp_net.get_action_probs(input_data)
        action_prob = action_prob.reshape(self.chessboard_height, self.chessboard_width)
        # action_prob = np.random.random((self.chessboard_height, self.chessboard_width))
        # action_prob = np.squeeze(action_prob, axis=0)
        # 将不符要求的点的概率设置为-1
        action_prob = action_prob.tolist()
        for i in range(len(action_prob)):
            for j in range(len(action_prob[i])):
                if 0 != chessboard_list[-1][i][j]:
                    action_prob[i][j] = -1

        return action_prob

    def simulation_once(self, chessboard_list, init_player):
        """
        从self.root开始进行一次MCTS的模拟
        :param chessboard_list: 待模拟的棋盘列表
        :param init_player: 初始玩家
        :return: 无
        """
        count_buff = 0

        node = self.root_node
        player = init_player  # 初始玩家为P1
        chessboard = copy.deepcopy(chessboard_list[-1])

        if len(chessboard_list) < self.chessboard_history_num:
            diff_value = self.chessboard_history_num - len(chessboard_list)
            # 填充空历史棋盘
            for _ in range(diff_value):
                empty_list = np.zeros(shape=(self.chessboard_height,
                                             self.chessboard_width),
                                      dtype=float).tolist()
                chessboard_list.insert(0, empty_list)

        if 0 == len(node.children):
            # 初次拓展
            first_action_prob = self.get_action_prob(chessboard_list)
            self.root_node.expand(first_action_prob)

        while True:
            # select
            node = node.select(self._c)

            MCTS.do_action(chessboard, node.action, player)  # 更新chessboard
            chessboard_list = self.renew_chessboard_list(chessboard_list, chessboard)  # 更新chessboard_list
            player = MCTS.reverse_player(player)  # 改变player

            # 绘制新模拟棋盘
            if self.if_print_sim_once:
                print_chessboard(chessboard)
                time.sleep(0.5)

            # 检测胜负
            count_buff += 1
            if count_buff > 15:
                winner = np.random.randint(0, 3)
                if winner == init_player:  # 初始玩家赢了
                    leaf_value = 1.0
                elif winner == 1 or winner == 2:  # 初始玩家输了
                    leaf_value = -1.0
                else:  # 平局
                    leaf_value = 0.0
                node.back_propagation(leaf_value)
                break
            else:
                # expand
                first_action_prob = self.get_action_prob(chessboard_list)
                node.expand(first_action_prob)

    def get_root_node_action_prob(self, chessboard, sim_player, temp=1e-3):
        """
        MCTS模拟self.simulation_num次，最终获得下一步决策的动作列表和概率列表
        :param chessboard:待MCTS的棋盘
        :param sim_player:模拟玩家
        :param temp:(0, 1]的温度值
        :return:action_list, probs的动作列表和概率列表
        """
        for i in range(self.simulation_num):
            print(str(i / self.simulation_num * 100) + "%已完成")
            sim_chessboard = copy.deepcopy(chessboard)
            self.simulation_once(sim_chessboard, sim_player)

        # 提取所有action及visits值
        action_list = []
        visits_list = []
        for child in self.root_node.children:
            action_list.append(child.action)
            visits_list.append(child._n)

        buff = 1.0 / temp * np.log(np.array(visits_list) + 1e-10)
        probs = np.exp(buff - np.max(buff))
        probs /= np.sum(probs)

        return action_list, probs











