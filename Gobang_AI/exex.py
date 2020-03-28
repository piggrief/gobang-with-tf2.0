# -*- coding:utf-8 -*-
from Gobang_AI.mcts import MCTS
from Gobang_AI.model import PolicyValueNet
import numpy as np
import copy
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt


# PVNet Test
p_v_net = PolicyValueNet(7, 7, 4)
model = p_v_net.cnn_net()

random_test_data_set =np.random.randint(0, 3, size=(1, 4, 7, 7), dtype=int).tolist()
loss_list = []
entropy_list = []
value_list = []

start_time = time()
for _ in range(32):
    random_train_data_set = np.random.randint(0, 3, size=(10, 4, 7, 7), dtype=int).tolist()
    random_train_policy = np.random.random((10, 7, 7)).tolist()
    random_train_value = np.random.randint(0, 3, size=(10, 1), dtype=int).tolist()

    action_probs, value = p_v_net.get_action_probs(random_test_data_set)
    loss, entropy = p_v_net.train_once(random_train_data_set, random_train_policy,
                                       random_train_value, 0.001)
    loss_list.append(loss)
    entropy_list.append(entropy)
    value_list.append(value.tolist()[0])
    print("Entropy:" + str(entropy))

stop_time = time()
spend_time = stop_time - start_time
print("平均一个局面的数据用时：" + str(spend_time / 32 / 10 * 1000) + "ms")

x = range(len(entropy_list))
# plt.plot(x, entropy_list)
# plt.show()


# tf.keras.utils.plot_model(model, \
#                           to_file='G:\\Program\\gobang-with-tf2.0\\figures\\model.png', \
#                           dpi=300,
#                           show_shapes=True)

# MCTS Test
search = MCTS(chessboard_height=7, chessboard_width=7,
              if_print_sim_once=False, simulation_num=5)
init_chessboard = copy.deepcopy(search.init_chessboard)
st = time()
action, probs = search.get_root_node_action_prob([init_chessboard], 1, temp=1e-3)
sst = time()
print("耗费：" + str(sst-st) + "s")

lis = [[2, 3], [3, 4]]
print(len(lis))
