#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from Gobang_AI.mcts import MCTS
import numpy as np
import copy
from time import time


search = MCTS(chessboard_height=7, chessboard_width=7,
              if_print_sim_once=False, simulation_num=300)
init_chessboard = copy.deepcopy(search.init_chessboard)
st = time()
action, probs = search.get_root_node_action_prob([init_chessboard], 1, temp=1e-3)
sst = time()
print("耗费：" + str(sst-st) + "s")

lis = [[2, 3], [3, 4]]
print(len(lis))
