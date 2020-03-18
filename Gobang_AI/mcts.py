#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import math
import copy


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
                 chessboard_history_num=4, _c=5):
        self.chessboard_height = chessboard_height
        self.chessboard_width = chessboard_width
        self.chessboard_history_num = chessboard_history_num
        self.chessboard = np.zeros(shape=(chessboard_height, chessboard_width), dtype=float)  # 0为没有，1为P1，2为P2
        self.chessboard = self.chessboard.tolist()
        self.root_node = MCTSTreeNode(None, (-1, -1), 1)
        self.chessboard_list = [(np.zeros(shape=(chessboard_height, chessboard_width), dtype=float)).tolist()]

        self._c = _c

    def renew_chessboard_list(self, chessboard):
        """
        用这次记录的棋盘动态更新棋盘列表
        :param chessboard: 这次棋盘
        :return:棋盘列表 (内部已经更新self.chessboard_list)
        """
        self.chessboard_list.append(copy.deepcopy(chessboard))
        diff_value = self.chessboard_history_num - len(self.chessboard_list)
        if diff_value > 0:
            # 填充空历史棋盘
            for _ in range(diff_value):
                empty_list = np.zeros(shape=(self.chessboard_height,
                                             self.chessboard_width),
                                      dtype=float).tolist()
                self.chessboard_list.insert(0, empty_list)
        elif diff_value < 0:
            self.chessboard_list = self.chessboard_list[-diff_value:]
        return self.chessboard_list

    def do_action(self, action, player):
        """
        执行玩家player落子action，更新self.chessboard
        :param action: 落子
        :param player: 玩家
        :return: self.chessboard
        """
        self.chessboard[action[0]][action[1]] = player
        return self.chessboard

    @staticmethod
    def reverse_player(player):
        if 1 == player:
            return 2
        else:
            return 1

    def get_action_prob(self, chessboard_list):
        """
        输入棋盘列表（n个历史棋盘）并通过价值策略网络生成下一步落子概率数组（h*w）
        :param chessboard_list: 棋盘列表（n个历史棋盘）
        :return: 下一步落子概率数组（h*w）
        """
        # 策略价值网络预测
        action_prob = np.random.random((self.chessboard_height, self.chessboard_width))
        # 将不符要求的点的概率设置为-1
        action_prob = action_prob.tolist()
        for i in range(len(action_prob)):
            for j in range(len(action_prob[i])):
                if 0 != chessboard_list[-1][i][j]:
                    action_prob[i][j] = -1

        return action_prob

    def simulation_once(self):
        count_buff = 0

        node = self.root_node
        init_player = 1
        player = init_player  # 初始玩家为P1

        self.renew_chessboard_list(self.chessboard)
        first_action_prob = self.get_action_prob(self.chessboard_list)
        # 初次拓展
        self.root_node.expand(first_action_prob)
        while True:
            # select
            node = node.select(self._c)

            self.do_action(node.action, player)  # 更新chessboard
            self.renew_chessboard_list(self.chessboard)  # 更新chessboard_list
            player = MCTS.reverse_player(player)  # 改变player

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
                first_action_prob = self.get_action_prob(self.chessboard_list)
                node.expand(first_action_prob)









