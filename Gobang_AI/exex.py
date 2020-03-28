#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#from Gobang_AI.mcts import MCTS
#from Gobang_AI.model import PolicyValueNet
import numpy as np
import copy
from time import time
import tensorflow as tf


# PVNet Test
p_v_net = PolicyValueNet(7, 7, 4)
model = p_v_net.cnn_net()
tf.keras.utils.plot_model(model, to_file='G:\\Program\\gobang-with-tf2.0\\figures\\model.png', dpi=300)

# MCTS Test
# search = MCTS(chessboard_height=7, chessboard_width=7,
#               if_print_sim_once=False, simulation_num=300)
# init_chessboard = copy.deepcopy(search.init_chessboard)
# st = time()
# action, probs = search.get_root_node_action_prob([init_chessboard], 1, temp=1e-3)
# sst = time()
# print("耗费：" + str(sst-st) + "s")

lis = [[2, 3], [3, 4]]
print(len(lis))
