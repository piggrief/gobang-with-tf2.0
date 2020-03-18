#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from Gobang_AI.mcts import MCTS

search = MCTS()
search.simulation_once()

lis = [[2, 3], [3, 4]]
print(len(lis))
