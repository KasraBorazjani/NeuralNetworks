import numpy as np
import pandas as pd
import adaline as ada
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_path, "Q3results")
plotnum = 1

x11 = np.random.normal(1, 0.5, size=(1000,2))
x11 = np.concatenate((x11,np.ones((1000,1))), axis=1)
x12 = np.random.normal(-1, 0.5, size=(10,2))
x12 = np.concatenate((x12,-1*np.ones((10,1))), axis=1)
x21 = np.random.normal(1, 0.5, size=(100,2))
x21 = np.concatenate((x21,np.ones((100,1))), axis=1)
x22 = np.random.normal(-1, 0.5, size=(100,2))
x22 = np.concatenate((x22,-1*np.ones((100,1))), axis=1)

data1 = np.concatenate((x11,x12), axis=0)
data2 = np.concatenate((x21,x22), axis=0)

newron = ada.adalineuron(data1)
neweron = ada.adalineuron(data2)

print("Metro-ing newron:")
newron.metro(learning_rate=0.1,threshold=0.2,epnum=100)
print("Newron weights:\n{}".format([newron.weights,newron.bias]))

print("Metro-ing neweron:")
neweron.metro(learning_rate=0.1,threshold=0.2,epnum=1000)
print("Neweron weights:\n{}".format([neweron.weights,neweron.bias]))

plotnum = newron.nevaluate(plotnum, save_path)
plotnum = neweron.nevaluate(plotnum, save_path)


print("Training newron:")
newron.train(learning_rate=0.1, threshold=0.2, gamma=1, epnum=100)

print("Training neweron:")
neweron.train(learning_rate=0.1, threshold=0.2, gamma=1, epnum=1000)


plotnum = newron.nevaluate(plotnum, save_path)
plotnum = neweron.nevaluate(plotnum, save_path)

