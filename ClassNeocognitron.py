#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:36:31 2017

@author: vadimkuzmin
"""

import numpy as np
from functions import f, f_v, filter3x3, filter5x5, filter7x7, filter_weight, psi, s_output
from patterns import S_1_train, S_2_train, S_3_train, S_4_train, n_s2, n_s3, n_s4

class Neocognitron:
    __private_filter3 = filter_weight(3)
    __private_filter5 = filter_weight(5)
    __private_filter7 = filter_weight(7)

    def __init__(self, gamma=np.array([0.9, 0.9, 0.9, 0.8]),
                 delta_bar=np.array([4.0, 4.0, 2.5, 1.0]), delta=np.array([0.9, 0.8, 0.7, 1.0]), r=np.array([1.7,4.0,1.5,1.0]),
                 alpha=np.array([10e4]*4)):
        # гиперпараметры
        self.gamma = gamma
        self.delta_bar = delta_bar
        self.delta = delta
        self.r = r
        self.alpha = alpha
        self.seed = np.array([[5,6],[5,6],
                 [7,6],[7,6],[7,6],
                 [5,6],
                 [7,6],
                 [6,7],
                 [6,5],
                 [5,5],[5,5],
                 [7,5],[7,5],
                 [7,7],
                 [6,6],[6,6],[6,6],
                 [6,5],
                 [6,7],[6,7],[6,7],
                 [5,7],
                 [7,7],[7,7],
                 [5,5],
                 [6,5],
                 [5,7],
                 [7,5],
                 [6,5],[6,5],
                 [5,5],
                 [5,7],
                 [6,5],[6,5],
                 [6,7]])

        self.C0 = np.array([[0.0] * 19] * 19)
        self.S1 = np.array([[[0.0] * 19] * 19] * 12)
        self.C1 = np.array([[[0.0] * 21] * 21] * 8)
        self.S2 = np.array([[[0.0] * 21] * 21] * 38)
        self.C2 = np.array([[[0.0] * 13] * 13] * 19)
        self.S3 = np.array([[[0.0] * 13] * 13] * 35)
        self.C3 = np.array([[[0.0] * 7] * 7] * 23)
        self.S4 = np.array([[[0.0] * 3] * 3] * 11)
        self.C4 = np.array([0.0] * 10)
        self.V1 = np.array([[0.0] * 19] * 19)
        self.V2 = np.array([[0.0] * 21] * 21)
        self.V3 = np.array([[0.0] * 13] * 13)
        self.V4 = np.array([[0.0] * 3] * 3)

        # определим матрицы весов для нашей нейронной сети
        self.a_S1 = np.array([[[0.0] * 3] * 3] * 12)  # для каждого из 12 паттернов матрица веса 3 на 3, в начале нули
        self.b_S1 = np.array([0.0] * 12)  # для каждого из 12 паттернов свой вес от тормозящего нейрона
        self.c_V1 = gamma[0] ** self.__private_filter3
        self.d_C1 = delta_bar[0] * delta[0] ** self.__private_filter3

        self.a_S2 = np.array([[[[0.0] * 5] * 5] * 8] * 38)  # для каждого из 38 приходится 8 матриц  5 на 5
        self.b_S2 = np.array([0.0] * 38)
        self.c_V2 = gamma[1] ** self.__private_filter5 # fixed
        self.d_C2 = delta_bar[1] * delta[1] ** self.__private_filter7  # fixed

        self.a_S3 = np.array([[[[0.0] * 5] * 5] * 19] * 35)  # для каждого из 35 приходится 19 матрицы 3 на 3
        self.b_S3 = np.array([0.0] * 35)
        self.c_V3 = gamma[2] ** self.__private_filter5
        self.d_C3 = delta_bar[2] * delta[2] ** self.__private_filter5  # fixed

        self.a_S4 = np.array([[[[0.0] * 5] * 5] * 23] * 11)  # для каждого из 11 23 матриц 5 на 5
        self.b_S4 = np.array([0.0] * 11)  # для каждого из 16 один тормозящий вес
        self.c_V4 = gamma[3] ** self.__private_filter5 # fixed
        self.d_C4 = delta_bar[3] * delta[3] ** self.__private_filter3  # fixed

        # схема перехода от S к С
        self.S1_to_C1 = np.array([1, 2, 1, 2, 1, 2, 1, 2])
        self.S2_to_C2 = np.array([1, 1, 1, 1, 4, 3, 3, 2, 1, 1, 2, 1, 5, 4, 1, 2, 2, 2, 1])
        self.S3_to_C3 = np.array([2, 3, 1, 1, 1, 1, 2, 2, 1, 3, 1, 3, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1])
        self.S4_to_C4 = np.array([1, 1, 1, 1, 2, 1, 1, 1, 1, 1])

    # очищает все нейроны
    # p - флаг очищения входного слоя
    def clear_net(self, p=1):
        if p == 1:
            self.C0.fill(0.0)
        self.S1.fill(0.0)
        self.C1.fill(0.0)
        self.S2.fill(0.0)
        self.C2.fill(0.0)
        self.S3.fill(0.0)
        self.C3.fill(0.0)
        self.S4.fill(0.0)
        self.C4.fill(0.0)
        self.V1.fill(0.0)
        self.V2.fill(0.0)
        self.V3.fill(0.0)
        self.V4.fill(0.0)

    def clear_weights(self):
        self.a_S1.fill(0.0)
        self.b_S1.fill(0.0)
        self.a_S2.fill(0.0)
        self.b_S2.fill(0.0)
        self.a_S3.fill(0.0)
        self.b_S3.fill(0.0)
        self.a_S4.fill(0.0)
        self.b_S4.fill(0.0)

    # заполним нейроны слоя V1
    def fill_V1(self):
        l = len(self.V1)
        F = filter3x3(self.C0,l)  # разбиваем на фильтры
        for i in range(l):
            for j in range(l):
                self.V1[i, j] = np.sqrt(f_v(self.c_V1, F[i, j]))

    # заполним S1
    def fill_S1(self):
        F1 = filter3x3(self.C0, 19)
        for k in range(len(self.S1)):
            for i in range(self.S1.shape[1]):
                for j in range(self.S1.shape[1]):
                    self.S1[k, i, j] = s_output(f(self.a_S1[k], F1[i, j]), self.V1[i, j], self.b_S1[k], self.r[0])

    # C1
    def fill_C1(self):
        F2 = np.array([filter3x3(self.S1[i], self.C1.shape[1]) for i in range(len(self.S1))]) # разбили каждый array по фильтру
        q = 0  # начальное смещение = 0
        for k in range(len(self.C1)):  # по всем плоскостям в С1, обход:
            for i in range(self.C1.shape[1]):
                for j in range(self.C1.shape[1]):
                    c_in = 0.0
                    for l in range(self.S1_to_C1[k]):  # l - сколько плоскостей видит в предыдущем слое
                        c_in += f(self.d_C1, F2[q + l, i, j])
                    self.C1[k, i, j] = psi(c_in)
            q += self.S1_to_C1[k]  # так как без повторений

    # заполним V2
    def fill_V2(self):
        self.V2.fill(0.0)
        l = len(self.V2)
        for k in range(len(self.C1)):  # по всем плоскостям в предыдущем слое, разбиваем их
            F3 = filter5x5(l, self.C1[k], 1)
            for i in range(l):
                for j in range(l):
                    self.V2[i, j] += f_v(self.c_V2, F3[i, j])

    # S2
    def fill_S2(self):
        F4 = np.array([filter5x5(self.S2.shape[1], self.C1[i], 1) for i in range(len(self.C1))])  # разбили каждый array по фильтру
        for k in range(len(self.S2)):
            for i in range(self.S2.shape[1]):
                for j in range(self.S2.shape[1]):
                    c_w = 0.0
                    for l in range(len(self.C1)):  # по всем плоскостям предыдущего слоя суммируем входы
                        c_w += f(self.a_S2[k, l], F4[l, i, j])
                    self.S2[k, i, j] = s_output(c_w, np.sqrt(self.V2[i, j]), self.b_S2[k], self.r[1])

    # C2
    def fill_C2(self):
        F5 = np.array([filter7x7(self.S2[i], self.C2.shape[1]) for i in range(len(self.S2))])  # разбили каждый array по фильтру
        q = 0
        for k in range(len(self.C2)):
            for i in range(self.C2.shape[1]):
                for j in range(self.C2.shape[2]):
                    c_in = 0.0
                    for l in range(self.S2_to_C2[k]):
                        c_in += f(self.d_C2, F5[q + l, i, j])
                    self.C2[k, i, j] = psi(c_in)
            q += self.S2_to_C2[k]  # так как no повторяющиеся

    # заполним V3
    def fill_V3(self):
        self.V3.fill(0.0)
        l = len(self.V3)
        for k in range(len(self.C2)):  # по всем плоскостям в предыдущем слое, разбиваем их
            F6 = filter5x5(l, self.C2[k], 1)
            for i in range(l):
                for j in range(l):
                    self.V3[i, j] += f_v(self.c_V3, F6[i, j])

    # S3
    def fill_S3(self):
        F7 = np.array([filter5x5(self.S3.shape[1], self.C2[i], 1) for i in range(len(self.C2))])  # разбили каждый array по фильтру
        for k in range(len(self.S3)):
            for i in range(self.S3.shape[1]):
                for j in range(self.S3.shape[2]):
                    c_w = 0.0
                    for l in range(len(self.C2)):  # по всем плоскостям предыдущего слоя суммируем входы
                        c_w += f(self.a_S3[k, l], F7[l, i, j])
                    self.S3[k, i, j] = s_output(c_w, np.sqrt(self.V3[i, j]), self.b_S3[k], self.r[2])

    # C3
    def fill_C3(self):
        F8 = np.array([filter5x5(self.C3.shape[1], self.S3[i], 2) for i in range(len(self.S3))])  # разбили каждый array по фильтру
        q = 0
        for k in range(len(self.C3)):
            for i in range(self.C3.shape[1]):
                for j in range(self.C3.shape[2]):
                    c_in = 0.0
                    for l in range(self.S3_to_C3[k]):
                        c_in += f(self.d_C3, F8[q + l, i, j])
                    self.C3[k, i, j] = psi(c_in)
            q += self.S3_to_C3[k]  # так как нет повторяющихся

    # заполним V4
    def fill_V4(self):
        self.V4.fill(0.0)
        l = len(self.V4)
        for k in range(len(self.C3)):  # по всем плоскостям в предыдущем слое, разбиваем их
            F9 = filter5x5(l, self.C3[k], 1)
            for i in range(l):
                for j in range(l):
                    self.V4[i, j] += f_v(self.c_V4, F9[i, j])

    # S3
    def fill_S4(self):
        F10 = np.array(
            [filter5x5(self.S4.shape[1], self.C3[i], 1) for i in range(len(self.C3))])  # разбили каждый array по фильтру
        for k in range(len(self.S4)):
            for i in range(self.S4.shape[1]):
                for j in range(self.S4.shape[2]):
                    c_w = 0.0
                    for l in range(len(self.C3)):  # по всем плоскостям предыдущего слоя суммируем входы
                        c_w += f(self.a_S4[k, l], F10[l, i, j])
                    self.S4[k, i, j] = s_output(c_w, np.sqrt(self.V4[i, j]), self.b_S4[k], self.r[3])

    # C4
    def fill_C4(self):
        q = 0
        for k in range(len(self.C4)):
            c_in = 0.0
            for l in range(self.S4_to_C4[k]):
                c_in += f(self.d_C4, self.S4[q + l])
            self.C4[k] = psi(c_in)
            q += self.S4_to_C4[k]

    def train_S1(self, alpha):
        for i in range(self.S1.shape[0]):
            self.a_S1[i] += alpha * np.multiply(self.c_V1, S_1_train[i])
            self.b_S1[i] += alpha * np.sqrt(f_v(self.c_V1, S_1_train[i])) #self.V1[9, 9]
            self.clear_net()

    def train_S2(self,alpha):
        for i in range(self.S2.shape[0]):  # для каждого array  в слое S2 = для каждого training pattern
            for k in range(n_s2[i]):
                self.C0[5:14, 5:14] = np.copy(S_2_train[k, i])
                self.fill_V1()
                self.fill_S1()
                self.fill_C1()
                self.fill_V2()
                for j in range(self.C1.shape[0]):  # для каждого array  в слое C1
                    self.a_S2[i, j] += alpha * np.multiply(self.c_V2, self.C1[j, 8:13, 8:13])
                self.b_S2[i] += alpha * np.sqrt(self.V2[10, 10])
                self.clear_net()

    def train_S3(self, alpha):
        for i in range(self.S3.shape[0]):  # для каждого array  в слое S3 = для каждого training pattern
            for k in range(n_s3[i]):
                self.C0 = np.copy(S_3_train[k, i])
                self.fill_V1()
                self.fill_S1()
                self.fill_C1()
                self.fill_V2()
                self.fill_S2()
                self.fill_C2()
                self.fill_V3()
                for j in range(self.C2.shape[0]):  # для каждого array  в слое C2
                    self.a_S3[i, j] += alpha * np.multiply(self.c_V3, self.C2[j, self.seed[i, 0] - 2:self.seed[i, 0] + 3,
                                                            self.seed[i, 1] - 2:self.seed[i, 1] + 3])
                self.b_S3[i] += alpha * np.sqrt(self.V3[self.seed[i, 0], self.seed[i, 1]])
                self.clear_net()

    def train_S4(self, alpha):
        for i in range(self.S4.shape[0]):  # для каждого array  в слое S4 = для каждого training pattern
            for k in range(n_s4[i]):
                self.C0 = np.copy(S_4_train[k, i])
                self.fill_V1()
                self.fill_S1()
                self.fill_C1()
                self.fill_V2()
                self.fill_S2()
                self.fill_C2()
                self.fill_V3()
                self.fill_S3()
                self.fill_C3()
                self.fill_V4()
                for j in range(self.C3.shape[0]):  # для каждого array  в слое C3
                    self.a_S4[i, j] += alpha * np.multiply(self.c_V4, self.C3[j, 1:6, 1:6])
                self.b_S4[i] += alpha * np.sqrt(self.V4[1, 1])
                self.clear_net()

    def train(self, alpha):
        self.clear_net()
        self.clear_weights()
        self.train_S1(alpha[0])
        self.train_S2(alpha[1])
        self.train_S3(alpha[2])
        self.train_S4(alpha[3])

    def predict(self,figure):
        self.clear_net()
        self.C0 = figure
        self.fill_V1()
        self.fill_S1()
        self.fill_C1()
        self.fill_V2()
        self.fill_S2()
        self.fill_C2()
        self.fill_V3()
        self.fill_S3()
        self.fill_C3()
        self.fill_V4()
        self.fill_S4()
        self.fill_C4()
        return np.argmax(self.C4)

    