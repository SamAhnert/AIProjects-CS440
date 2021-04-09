# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        super(NeuralNet, self).__init__()
        self.lrate = lrate
        self.loss_fn = loss_fn
        self.model_convoluted = nn.Sequential(nn.Conv2d(3,6,5), nn.LeakyReLU(), nn.MaxPool2d(2,2),
                                nn.Conv2d(6,16,5), nn.LeakyReLU(), nn.MaxPool2d(2,2) )
        #16 * 5 * 5 = outputs of convd2 * 
        self.model_linear = nn.Sequential( nn.Linear(16*5*5,32), nn.ReLU(),
                            nn.Linear(32,out_size), nn.ReLU(), nn.LogSoftmax() ) 
        self.optimizer = optim.SGD(self.parameters(), lr=self.lrate,momentum=.8,weight_decay=5e-5)

    def forward(self, x):
        #configure tensor to fit our conv layer
        x = x.view(-1, 3, 32, 32)
        x = self.model_convoluted(x)
        #configure conv output to fit our linear layer
        #add an extra dim for batch size x 400
        x = x.view(-1, 16*5*5)
        y = self.model_linear(x)
        return y

    def step(self, x,y):
        self.model_convoluted.zero_grad()
        self.model_linear.zero_grad()
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred,y)
        
        loss.backward()

        self.optimizer.step()

        return loss

def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    train_set = (train_set - torch.mean(train_set)) / torch.std(train_set)
    dev_set = (dev_set - torch.mean(dev_set)) / torch.std(dev_set)

    net = NeuralNet(0.01, nn.CrossEntropyLoss(), 3072, 2)

    loss = []

    #Randomly start somewhere in the training_data
    #We want to stay on intervals of batch_size since if we input 7499
    #into the below algorithm we only train our model w one piece of data
    index = np.random.randint(len(train_set) // batch_size)

    for ii in range(0,n_iter):
        #start index of data to train
        start = index * batch_size
        end = (start+1)*batch_size
        if (ii+1)*start >= len(train_set):
            #adjust end
            end = len(train_set) - 1
            this_loss = net.step(train_set[start: end], train_labels[start: end])
            #reset to front of data
            index=0
        else:
            this_loss = net.step(train_set[start:end], train_labels[start:end])
            index+=1
        loss.append(this_loss)
        # print(ii, loss[ii])
            
    y_hat = np.argmax(net.forward(dev_set).detach().numpy(), axis=1)

    return loss,y_hat,net
