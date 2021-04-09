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
within this file and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.lrate = lrate
        self.loss_fn = loss_fn
        self.model = nn.Sequential(nn.Linear(in_size,32), nn.ReLU(), nn.Linear(32,out_size), nn.LogSoftmax())
        self.optimizer = optim.SGD(self.parameters(), lr=self.lrate,momentum=.8)

    def set_parameters(self, params):
        """ Sets the parameters of your network.

        @param params: a list of tensors containing all parameters of the network
        """
        self.parameters = params
    
    def get_parameters(self):
        """ Gets the parameters of your network.

        @return params: a list of tensors containing all parameters of the network
        """
        return self.parameters()

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        y = self.model(x)
        return y

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        self.model.zero_grad()
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred,y)
        
        loss.backward()

        self.optimizer.step()

        # optimizer.step()
        return loss


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    train_set = (train_set - torch.mean(train_set)) / torch.std(train_set)
    dev_set = (dev_set - torch.mean(dev_set)) / torch.std(dev_set)

    net = NeuralNet(0.01, nn.CrossEntropyLoss(), 3072, 2)

    loss = []

    #We want to stay on intervals of batch_size since if we input 7499
    #into the below algorithm we only train our model w one piece of data
    index = np.random.randint(len(train_set) // batch_size)
    print(index)
    print(len(train_set))
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
            
    y_hat = np.argmax(net.forward(dev_set).detach().numpy(), axis=1)

    return loss,y_hat,net
