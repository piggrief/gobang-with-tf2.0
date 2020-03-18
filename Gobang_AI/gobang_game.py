#!/usr/bin/env python 
# -*- coding:utf-8 -*-


def print_chessboard(chessboard_to_print):
    for i in range(0, 2 * chessboard_to_print[0].__len__() + 2):
        print('-', end='')
    print()
    for i in range(len(chessboard_to_print)):
        print("|", end='')
        for j in range(len(chessboard_to_print[i])):
            if 2 == chessboard_to_print[i][j]:
                print("O ", end='')
            elif 1 == chessboard_to_print[i][j]:
                print("X ", end='')
            else:
                print("  ", end='')
        print('|')
    for i in range(0, 2 * chessboard_to_print[0].__len__() + 2):
        print('-', end='')
    print()
