#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:13:13 2017

@author: vadimkuzmin
"""
import numpy as np


def f(a, b):
    return np.sum(np.multiply(a, b))  # умножаем поэлементно и складываем элементы


# поэлементное умножение матриц, где матрица B возведена в квадрат поэлементно и сложение всех элементов
def f_v(a, b):
    return np.sum(np.multiply(a, np.multiply(b, b)))


# indicator
def phi(x):
    if x >= 0.0:
        return x
    else:
        return 0.0


# for C-cell output
def psi(l):
    return phi(l)/(1+phi(l))


def s_output(c_weighted, v_input, b_weight, r=1.0):
    return r*phi((1+c_weighted)/(1+r/(1+r)*b_weight*v_input)-1)


# taxicab metric weight  - 1 matrix nxn
def filter_weight(n):
    u = np.array([[0.0] * n] * n)
    middle = int(n/2)
    for i in range(n):
        for j in range(n):
            u[i, j] = (1+np.fabs(i - middle)+np.fabs(j - middle))-1
    return u


def filter3x3(M,k):
    'k - размер матрицы, для которой фильтр'
    n = len(M)
    F = np.array([[[[0.0]*3]*3]*k]*k)
    L = np.array([[0.0] * (k+2)] * (k+2))
    L[int((k-n+2)/2):n+int((k-n+2)/2),int((k-n+2)/2):n+int((k-n+2)/2)] = M
    for i in range(k):
        for j in range(k):
            F[i,j] = L[i:i+3,j:j+3]
    return F


def filter5x5(n,M,s):
    'n - размер для которой разбиваем, s - шаг'
    F = np.array([[[[0.0]*5]*5]*n]*n)
    z = int(n/2)*s+2-int(len(M)/2)
    L = np.array([[0.0] * (len(M)+z*2)] * (len(M)+z*2))
    L[z:len(M)+z,z:len(M)+z] = M
    for i in range(n):
        for j in range(n):
            F[i,j] = L[i*s:i*s+5,j*s:j*s+5]
    return F


def filter7x7(M, k):
    'k - размер матрицы, для которой фильтр'
    n = len(M)
    F = np.array([[[[0.0]*7]*7]*k]*k)
    L = np.array([[0.0] * (n+10)] * (n+10))
    L[5:len(M)+5,5:len(M)+5] = M
    for i in range(k):
        for j in range(k):
            F[i,j] = L[i*2:i*2+7,j*2:j*2+7]
    return F