#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YH
"""

import numpy as np 
from numpy.random import shuffle
from scipy.linalg import svd
from numpy.linalg import inv

#read data
def readData(dataset):   
    res = []
    f = open(dataset,'r')
    lines = f.readlines()
    for line in lines:
        values = line.split()
        values_integer = []
        for val in values:
            values_integer.append(int(val))
        res.append(values_integer)
    f.close()
    return np.asarray(res)

#read data
points2D = readData('/Users/2Dpoints.txt')
points3D_part1 = readData('/Users/3Dpoints_part1.txt')
points3D_part2 = readData('/Users/3Dpoints_part2.txt')

len(points2D[0])

#data normalization, H2D and H3D
def normalize(data):   
    size = len(data[0])
    h_matrix = np.zeros((size+1,size+1))
    h_matrix[size][size] = 1
    for i in range(0,size):
        h_matrix[i][size] = np.mean(data[:,i])
    sum = 0
    for i in range(len(data)):
        d = 0
        for j in range(0,size):
            d = d + (data[i][j] - h_matrix[j][size])**2
        sum = sum + np.sqrt(d)
    sd = sum/(len(data))
    for i in range(0,size):
        h_matrix[i][i] = 1/sd
        h_matrix[i][size] = -h_matrix[i][size]/sd
    return h_matrix
    
points2D_H2D = normalize(points2D)
points3D_part1_H3D = normalize(points3D_part1)
points3D_part2_H3D = normalize(points3D_part2)

#calculate m',M' matrix 
def normalized_Matrix(data,h):
    append = np.ones((len(data),1))
    res = np.hstack((data,append))
    res = res.transpose()
    res = np.matmul(h,res)
    res = res.transpose()
    return res
    
points2D_new = normalized_Matrix(points2D,points2D_H2D)
points3D_part1_new = normalized_Matrix(points3D_part1,points3D_part1_H3D) 
points3D_part2_new = normalized_Matrix(points3D_part2,points3D_part2_H3D) 


#split the data into two halves, one for train and the other for test
data_all = np.hstack((points2D_new,points3D_part1_new))
data_all = np.hstack((data_all,points3D_part2_new))

data_all = np.hstack((data_all,points2D))
data_all = np.hstack((data_all,points3D_part1))
data_all = np.hstack((data_all,points3D_part2))
shuffle(data_all)
train_len = int(len(data_all)/2)
train, test = data_all[:train_len,:],data_all[train_len:,:]
test1 = test

#generate the A' matrix
#need to be checked later on
def generate_A_norm_Matrix(data):
    res = np.zeros((len(data)*2,12))
    count_row = 0
    for i in range(len(data)):
        u = data[i][0]
        v = data[i][1]
        res[count_row][0] = data[i][3]
        res[count_row][1] = data[i][4]
        res[count_row][2] = data[i][5]
        res[count_row][3] = data[i][6]
        res[count_row][8] = -u*data[i][3]
        res[count_row][9] = -u*data[i][4]
        res[count_row][10] = -u*data[i][5]
        res[count_row][11] = -u
        count_row = count_row + 1
        res[count_row][4] = data[i][3]
        res[count_row][5] = data[i][4]
        res[count_row][6] = data[i][5]
        res[count_row][7] = data[i][6]
        res[count_row][8] = -v*data[i][3]
        res[count_row][9] = -v*data[i][4]
        res[count_row][10] = -v*data[i][5]
        res[count_row][11] = -v
        count_row = count_row + 1
    return res

#train A' matrix
train_A_norm = generate_A_norm_Matrix(train)


#perform SVD on A' matrix
U, s, VT = svd(train_A_norm)
V = VT.transpose()
v_norm = V[:,len(V)-1]
square = v_norm[9]**2 + v_norm[10]**2 + v_norm[11]**2
alpha = 1/np.sqrt(square)

#solve the P' matrix
v_norm = v_norm*alpha
P_norm = np.zeros((3,4))
P_norm[0,:] = v_norm[0:4]
P_norm[1,:] = v_norm[4:8]
P_norm[2,:] = v_norm[8:12]

#recover the P matrix
P = np.matmul(inv(points2D_H2D),P_norm)
P = np.matmul(P,points3D_part1_H3D)
np.set_printoptions(precision=4,suppress=True)
print("P matrix")
print(P)

#calculate u0, v0, fu and fv
u0 = P[0][0]*P[2][0] + P[0][1]*P[2][1] + P[0][2]*P[2][2]
v0 = P[1][0]*P[2][0] + P[1][1]*P[2][1] + P[1][2]*P[2][2]
fu = np.sqrt(P[0][0]*P[0][0] + P[0][1]*P[0][1] + P[0][2]*P[0][2] - u0*u0)
fv = np.sqrt(P[1][0]*P[1][0] + P[1][1]*P[1][1] + P[1][2]*P[1][2] - v0*v0)
print("u0: %.6f" % u0)
print("v0: %.6f" % v0)
print("fu: %.6f" % fu)
print("fv: %.6f" % fv)

#error analysis
ones = np.ones((len(test),1))
test_3D = test[:,13:16]
points3D = np.hstack((test_3D,ones))
test_estimated = np.matmul(P,points3D.transpose())
test_estimated = test_estimated.transpose()
error = 0
for i in range(len(test_estimated)):
    error = error + np.sqrt((test_estimated[i][0]/test_estimated[i][2]-test[i][11])**2 + (test_estimated[i][1]/test_estimated[i][2]-test[i][12])**2)
error = error/len(test_estimated)
print("error: %.6f" % error)
