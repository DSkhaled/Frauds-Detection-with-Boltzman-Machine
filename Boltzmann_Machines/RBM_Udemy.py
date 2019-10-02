# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:03:17 2019

@author: K6433702
"""

#%reset -f
#%clear

# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the Dataset
movies  = pd.read_csv("ml-1m/movies.dat",sep='::',header=None,engine='python',encoding='Latin-1')
users = pd.read_csv("ml-1m/users.dat",sep='::',header=None,engine='python',encoding='Latin-1')
ratings  = pd.read_csv("ml-1m/ratings.dat",sep='::',header=None,engine='python',encoding='Latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv("ml-100k/u1.base",delimiter='\t',header=None)
training_set = np.array(training_set,dtype = 'int64')
test_set = pd.read_csv("ml-100k/u1.test",delimiter='\t',header=None)
test_set = np.array(test_set,dtype='int64')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1,nb_users+1):
        id_movies = data[data[:,0]==id_users][:,1]
        id_ratings = data[data[:,0]==id_users][:,2]
        ratings = np.zeros(nb_movies,dtype='int64')
        ratings[id_movies-1]= id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# converting the ratings into binary ratings 1 (liked) or 0 (disliked)

# Training set
for elt in range(6):
    if elt == 0:
        training_set[training_set == elt] = -1
    elif elt in [1,2]:
        training_set[training_set == elt] = 0
    elif elt in [3,4,5]:
        training_set[training_set == elt] = 1
# Test set
for elt in range(6):
    if elt == 0:
        test_set[test_set == elt] = -1
    elif elt in [1,2]:
        test_set[test_set == elt] = 0
    elif elt in [3,4,5]:
        test_set[test_set == elt] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self,nv,nh):
        self.W = torch.randn(nh , nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)
        
    def sample_h(self,x):
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
        
    def sample_v(self,y):
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self,v0,vk,ph0,phk):
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.b += torch.sum((v0-vk),0)
        self.a += torch.sum((ph0-phk),0)

nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv,nh)

"""============================= Evaluating the Boltzmann Machine ============================"""

""" ************************************ Average Distance ************************************"""
# Training the RBM
nb_epochs = 10
for epoch in range(1,nb_epochs+1):
    train_loss = 0
    s = 0.
    for id_user in range(0,nb_users-batch_size,batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk,_ =rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(0,nb_users):
        v = training_set[id_user:id_user+1]
        vt = test_set[id_user:id_user+1]
        if len(vt[vt>=0]) > 0:
            _,h = rbm.sample_h(v)
            _,v = rbm.sample_v(h)
            test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
            s += 1.
print('test loss: '+str(test_loss/s))
""" ******************************************************************************************"""

""" ****************************************** RMSE ******************************************"""
# Training the RBM
nb_epochs = 10
for epoch in range(1,nb_epochs+1):
    train_loss = 0
    s = 0.
    for id_user in range(0,nb_users-batch_size,batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk,_ =rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss += np.sqrt(torch.mean((v0[v0 >= 0] - vk[v0 >= 0])**2))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(0,nb_users):
        v = training_set[id_user:id_user+1]
        vt = test_set[id_user:id_user+1]
        if len(vt[vt>=0]) > 0:
            _,h = rbm.sample_h(v)
            _,v = rbm.sample_v(h)
            test_loss += np.sqrt(torch.mean((vt[vt >= 0] - v[vt >= 0])**2))
            s += 1.
print('test loss: '+str(test_loss/s))
""" ******************************************************************************************"""