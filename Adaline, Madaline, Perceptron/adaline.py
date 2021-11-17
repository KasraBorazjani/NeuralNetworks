import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

class adalineuron():
    def __init__(self, dataset):
        self.traindata = dataset
        self.weightnum = len(dataset[0]) - 1

    def train(self, learning_rate, threshold, gamma, epnum):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.gamma = gamma
        self.weights = np.zeros((1,self.weightnum))
        self.bias = 0
        self.epnum = 1
        while(True):
            eploss = 0
            for data in self.traindata:
                x = data[:self.weightnum]
                target = data[self.weightnum]
                net = np.matmul(x, self.weights.T) + self.bias
                h = self.actifunction(net, self.gamma)
                dataloss = 0.5*(target - h)**2
                eploss += dataloss
                self.upweight(h, x, target)

                
            eploss /= len(self.traindata)
            
            if(self.epnum == epnum):
                print("epoch number {} loss = {}".format(self.epnum, eploss))
                break
            self.epnum += 1
        
        print("Successfully trained in {} epochs!".format(self.epnum))
    

    def actifunction(self, net, gamma):
        return math.tanh(gamma*net)

    def upweight(self, h, x, target):
        self.weights += self.learning_rate*(1-h**2)*self.gamma*(target - h)*x
        self.bias += self.learning_rate*(1-h**2)*self.gamma*(target - h)

    def metro(self, learning_rate, threshold, epnum):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.weights = np.zeros((1,self.weightnum))
        self.bias = 0
        self.epnum = 1
        while(True):
            eploss = 0
            for data in self.traindata:
                x = data[:self.weightnum]
                target = data[self.weightnum]
                net = np.matmul(x, self.weights.T) + self.bias
                # h = self.nactifunction(net)
                dataloss = 0.5*(target - net)**2
                eploss += dataloss
                self.highweight(net, x, target)
            
            eploss /= len(self.traindata)
            
            if(self.epnum == epnum):
                print("epoch number {} loss = {}".format(self.epnum, eploss))
                break
            self.epnum += 1
        
        print("Successfully trained in {} epochs!".format(self.epnum))
    

    def nactifunction(self, net):
        if(net >= 0):
            return 1
        else:
            return -1 

    def highweight(self, net, x, target):
        self.weights += self.learning_rate*(target - net)*x
        self.bias += self.learning_rate*(target - net)
    

    def nevaluate(self, plotnum, save_path):

        x1 = [i[0] for i in self.traindata if i[2]==1]
        x2 = [i[0] for i in self.traindata if i[2]==-1]
        y1 = [i[1] for i in self.traindata if i[2]==1]
        y2 = [i[1] for i in self.traindata if i[2]==-1]

        tx3 = np.linspace(-3,3,100)
        ty3 = -1*self.weights[0][0]/self.weights[0][1]*tx3 - self.bias/self.weights[0][1]

        fig = plt.figure()

        plt.scatter(x1, y1, color='r', label='class 1')
        plt.scatter(x2, y2, color='b', label='class 2')
        

        plt.plot(tx3, ty3, color='g', label='Split Line')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid()
        plt.title('Scatter Plot')
        fig.savefig(os.path.join(save_path, 'Q3P{}.png'.format(plotnum)), transparent = True)
        plotnum += 1

        return plotnum
    
            

            


