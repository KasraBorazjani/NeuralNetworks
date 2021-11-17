import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

class madalinetwork():
    def __init__(self, dataset, inSize, hiddenSize, outSize):
        self.traindata = dataset
        self.weightnum = len(dataset[0]) - 1
        self.insize = inSize
        self.hiddensize = hiddenSize
        self.outsize = outSize
        self.hiddenweights = np.random.normal(1, 0.005, (self.insize, self.hiddensize))
        self.hiddenbias = np.random.normal(1, 0.005,(1,self.hiddensize))
        self.outweights = 1/self.hiddensize*np.ones((self.hiddensize,self.outsize))
        self.outbias = (self.hiddensize-1)/self.hiddensize*np.ones(self.outsize)

    def train(self, learning_rate, epnum):
        self.learning_rate = learning_rate
        self.epnum = 1
        while(True):
            eploss = 0
            for data in self.traindata:
                x = data[:self.weightnum]
                target = data[self.weightnum]
                net1 = np.dot(x, self.hiddenweights) + self.hiddenbias
                h = self.actifunction(net1)
                net2 = np.dot(h, self.outweights) + self.outbias
                h2 = self.actifunction(net2)
                if (target!=h2):
                    if(target==1):
                        closest = np.argmin(np.abs(net1))
                        self.hiddenbias[0][closest] += self.learning_rate*(1-net1[0][closest])
                        for i in range(self.insize):
                            self.hiddenweights[i][closest] += self.learning_rate*(1-net1[0][closest])*x[i]
                    
                    elif(target==-1):
                        for k in range(self.hiddensize):
                            if(net1[0][k]>0):
                                self.hiddenbias[0][k] += self.learning_rate*(-1 - net1[0][k])
                                for j in range(self.insize):
                                    self.hiddenweights[j][k] += self.learning_rate*(-1-net1[0][k])*x[j]
                
                eploss += 0.5*(target - net2)**2
                
            eploss /= len(self.traindata)
            
            if(self.epnum == epnum):
                print("epoch number {} loss = {}".format(self.epnum, eploss))
                break
            self.epnum += 1
        
        print("Successfully trained in {} epochs!".format(self.epnum))
    

    def actifunction(self, net):
        return np.sign(net)
    

            

            


