import numpy as np
import pandas as pd
import madaline as maddy
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_path, "Q4results")
dataset_path = os.path.join(current_path, "madaline.csv")
fulldata = np.genfromtxt(dataset_path, delimiter=",")
plotnum = 1

for i in range(len(fulldata)):
    if fulldata[i][2] == 0:
        fulldata[i][2] = -1

newetwork = maddy.madalinetwork(fulldata, 2, 8, 1)
newetwork.train(learning_rate=0.1, epnum=2000)


neweights = newetwork.hiddenweights
nebias = newetwork.hiddenbias


x1 = [fulldata[j][0] for j in range(len(fulldata)) if fulldata[j][2]==1]
x2 = [fulldata[j][0] for j in range(len(fulldata)) if fulldata[j][2]==0]
y1 = [fulldata[i][1] for i in range(len(fulldata)) if fulldata[i][2]==1]
y2 = [fulldata[i][1] for i in range(len(fulldata)) if fulldata[i][2]==0]


plt.scatter(x1,y1, color='r')
plt.scatter(x2,y2, color='g')

x3 = np.linspace(-2, 2, 100)
for i in range(8):
    plt.plot(x3, -1*neweights[0][i]/neweights[1][i]*x3 - nebias[0][i]/neweights[1][i], color='b')
plt.xlim((-1.6,1.6))
plt.ylim((-1.6,1.6))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Result for {} lines and {} epochs'.format(8,2000))
plt.savefig(os.path.join(save_path, 'Q4P{}.png'.format(9)))
plt.show()