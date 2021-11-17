import numpy as np
import pandas as pd
import math
import os

class percepneuron():
    def __init__(self, dataset):
        new_dataset = dataset[1:]
        dataset_id = np.arange(len(new_dataset))
        self.traindatalen = int(0.75*len(new_dataset))
        traindata_id = np.random.choice(dataset_id, self.traindatalen, replace=False)
        self.traindata = new_dataset[traindata_id]
        testdata_id = np.setdiff1d(dataset_id, traindata_id)
        self.testdata = new_dataset[testdata_id]
        print("testdatalen = {}\ntraindatalen = {} \n".format(len(self.testdata), len(self.traindata)))
        self.weightnum = len(new_dataset[0]) - 1

    def train(self, learning_rate, threshold):
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
                h = self.actifunction(net)
                dataloss = h - target
                eploss += abs(dataloss)
                if(dataloss != 0):
                    self.weights += self.learning_rate*target*x
                    self.bias += self.learning_rate*target
                
            
            #print("epoch number {} loss = {}".format(self.epnum, eploss))
            if(eploss == 0):
                break
            self.epnum += 1
        
        print("Successfully trained in {} epochs!".format(self.epnum))
    

    def actifunction(self, net):
        if(net>self.threshold):
            return 1
        elif(net<=self.threshold and net>=-1*self.threshold):
            return 0
        else:
            return -1 

    
    def test(self):
        test_results = []
        for data in self.testdata:
            x = data[:self.weightnum]
            net = np.matmul(x, self.weights.T)
            h = self.actifunction(net)
            test_results.append(h)
        
        return test_results

    def evaluate(self, test_results):
        correct = 0
        for i in range(len(self.testdata)):
            if(self.testdata[i][self.weightnum] == test_results[i]):
                correct += 1
        accuracy = correct/len(self.testdata)*100
        print("Accuracy is {} percent".format(accuracy))
    
            

            


