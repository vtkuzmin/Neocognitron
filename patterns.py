#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:17:06 2017

@author: vadimkuzmin
"""

import numpy as np
from PIL import Image

# patterns for S1 level
S_1_train = np.array([[[0]*3]*3]*12)
S_1_train[0,1,:] = 1

S_1_train[1,1,:2] = 1
S_1_train[1,0,2] = 1

S_1_train[2,1,1:] = 1
S_1_train[2,2,0] = 1

S_1_train[3,2,0] = 1
S_1_train[3,1,1] = 1
S_1_train[3,0,2] = 1

S_1_train[4,2,0] = 1
S_1_train[4,:2,1] = 1

S_1_train[5,0,2] = 1
S_1_train[5,1:,1] = 1

S_1_train[6,:,1] = 1

S_1_train[7,0,0] = 1
S_1_train[7,1:,1] = 1

S_1_train[8,2,2] = 1
S_1_train[8,:2,1] = 1

S_1_train[9,0,0] = 1
S_1_train[9,1,1] = 1
S_1_train[9,2,2] = 1

S_1_train[10,1,:2] = 1
S_1_train[10,2,2] = 1

S_1_train[11,1,1:] = 1
S_1_train[11,0,0] = 1

n_s2 = np.array([4,4,4,4,4,4,2,3,4,2,3,4,2,4,4,4,1,3,5,4,3,3,4,3,3,4,3,3,4,3,3,3,4,3,3,1,2,2])

S_2_train = np.array([[[[0.0]*9]*9]*38]*5)
for i in range(38):
    for k in range(n_s2[i]):
        image = Image.open("Training patterns/S_2_{}_{}.png".format(i+1, k+1))
        image = np.array(image)[:,:,0]
        image[np.where(image == [0])] = [1]
        image[np.where(image == [255])] = [0]
        S_2_train[k,i,:,:]= image

n_s3 = np.array([2,3,2,3,1,2,2,2,2,2,2,3,2,2,2,1,1,2,2,1,2,1,4,4,2,2,2,1,3,2,2,3,1,2,3])

S_3_train = np.array([[[[0.0]*19]*19]*35]*4)
for i in range(35):
    for k in range(n_s3[i]):
        image = Image.open("Training patterns/S_3_{}_{}.png".format(i+1, k+1))
        image = np.array(image)[:,:,0]
        image[np.where(image == [0])] = [1]
        image[np.where(image == [255])] = [0]
        S_3_train[k, i,:,:]= image

n_s4 = np.array([2,3,2,2,3,4,1,2,2,2,2])

S_4_train = np.array([[[[0.0]*19]*19]*11]*4)
for i in range(11):
    for k in range(n_s4[i]):
        image = Image.open("Training patterns/S_4_{}_{}.png".format(i+1, k+1))
        image = np.array(image)[:,:,0]
        image[np.where(image == [0])] = [1]
        image[np.where(image == [255])] = [0]
        S_4_train[k,i,:,:]= image

S_test = np.array([[[0.0]*19]*19]*10)
for i in  range(10):
    image = Image.open("Training patterns/S_test_{}.png".format(i))
    image = np.array(image)[:,:,0]
    image[np.where(image == [0])] = [1]
    image[np.where(image == [255])] = [0]
    S_test[i,:,:]= image