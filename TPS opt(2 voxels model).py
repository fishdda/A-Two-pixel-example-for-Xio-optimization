import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom as dicom
import math

## This script is to simulate the optimization
## process of dose engine in Monaco TPS


## This is a two voxel model of patients ##


## Target and Normal tissue Goals ##
D_g = np.array([[64], [0]])

T = np.array([[16,16,16,16],[15,0,17,0]])
weight_init = np.ones([4,1])
# calculate dose
D_c = np.dot(T,weight_init)
                        
# calculate gradient
G_OB = 2*np.dot((D_c-D_g).T,(np.dot(T,np.eye(4))))

# Normalize gradient
G_OB_N_init = (G_OB/np.sqrt(np.sum(G_OB*G_OB))).T

# weight -= i*G_OB_N.T
## Part 1: Physical Optimization ##
iteration = 18
OBJ_plot,D_matrix = [],[[],[]]
for j in range(iteration):
    
    L = np.arange(0,1.5*(2/3)**math.log(j+1),0.05/(j+1))
    weight_mat = np.zeros([weight_init.shape[0],len(L)])
    D_c_mat = np.zeros([D_g.shape[0],len(L)])
    OBJ_mat = np.zeros([1,len(L)])
    G_OB_mat = np.zeros([G_OB_N_init.shape[0],len(L)])
    weight = weight_init
    G_OB_N = G_OB_N_init
    for i in range(len(L)):
        
        # update weights
        weight_ = weight - L[i]*G_OB_N

        # calculate dose
        D_c_ = np.dot(T,weight_)
                                
        # calculate gradient
        G_OB_ = 2*np.dot((D_c_-D_g).T,(np.dot(T,np.eye(4))))

        # Normalize gradient
        G_OB_N_ = G_OB_/np.sqrt(np.sum(G_OB_*G_OB_))

        # calculate object
        OBJ_mat[:,i] = np.sum((D_c_-D_g)*(D_c_-D_g))
        D_c_mat[:,i] = np.reshape(D_c_,[D_c.shape[0],])
        weight_mat[:,i] = np.reshape(weight_,[weight.shape[0],])
        G_OB_mat[:,i] = np.reshape(G_OB_N_,[G_OB_N.shape[0],])
    # To find the minimal OBJ index
    min_index = np.where(OBJ_mat== np.min(OBJ_mat))
    print('minimum OBJ ={}'.format(OBJ_mat[min_index]))
    print('{}th iter, the Dose matrix:{}'.format(j,D_c_mat[:,min_index[1]]))
    print('{}th iter, the Gradient:{}'.format(j,G_OB_mat[:,min_index[1]]))
    OBJ_plot.append(OBJ_mat[min_index][0])
    D_matrix[0].append(D_c_mat[:,min_index[1]][0][0])
    D_matrix[1].append(D_c_mat[:,min_index[1]][1][0])
    # Update the weights
    weight_init = np.reshape(weight_mat[:,min_index[1]],[weight.shape[0],1])
    G_OB_N_init = np.reshape(G_OB_mat[:,min_index[1]],[G_OB_N.shape[0],1])
    

plt.figure(1)
plt.plot(OBJ_plot,'r-*')
plt.xlabel('iteration')
plt.ylabel('OBJ function')
plt.grid()
plt.show()

plt.figure(2)
x=np.array([40,70])
y=np.array([-1,20])
z=np.array([[40,70],[-1,20]])
plt.xlim(40,70)
plt.ylim(-1.0,20)
plt.contourf(x,y,z,cmap='brg')
plt.plot(D_matrix[0],D_matrix[1],'b*')
plt.title('Contour plot of convergence(Target Goal:64Gy,Normal Goal:0Gy)')
plt.xlabel('Target Dose(Gy)')
plt.ylabel('Normal Tissue Dose(Gy)')
plt.show()

## Part 2: Biological Optimization ##


## plot contour plot to illustrate the gradient descent direction






## initial value of optimization ##


