import numpy as np
import pandas as pd
import perceptron as pcp
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_path, "Q2results")
dataset_path = os.path.join(current_path, "perceptron.csv")
fulldata = np.genfromtxt(dataset_path, delimiter=",")
plotnum = 1

''' Part 1 '''

x1 = [i[0] for i in fulldata if i[2]==1]
x2 = [i[0] for i in fulldata if i[2]==-1]
y1 = [i[1] for i in fulldata if i[2]==1]
y2 = [i[1] for i in fulldata if i[2]==-1]
plt.scatter(x1, y1, color='r', label='class 1')
plt.scatter(x2, y2, color='b', label='class 2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid()
plt.title('Scatter Plot')
plt.savefig(os.path.join(save_path, 'Q2P{}.png'.format(plotnum)), transparent = True)
plt.show()

plotnum += 1

''' Part 2'''

newron = pcp.percepneuron(fulldata)
newron.train(learning_rate=0.1, threshold=0.3)


''' Part 3 '''

def nevaluate(newron, plotnum):

    test_Res = newron.test()
    newron.evaluate(test_Res)

    fig = plt.figure()
    tx1 = [newron.testdata[i][0] for i in range(len(newron.testdata)) if newron.testdata[i][2]!=test_Res[i]]
    tx2 = [newron.testdata[i][0] for i in range(len(newron.testdata)) if newron.testdata[i][2]==test_Res[i]]
    ty1 = [newron.testdata[i][1] for i in range(len(newron.testdata)) if newron.testdata[i][2]!=test_Res[i]]
    ty2 = [newron.testdata[i][1] for i in range(len(newron.testdata)) if newron.testdata[i][2]==test_Res[i]]
    plt.scatter(tx1, ty1, color='r', label='Wrong Label')
    plt.scatter(tx2, ty2, color='g', label='Correct Label')
    tx3 = np.linspace(-2,2,100)
    ty3 = -1*newron.weights[0][0]/newron.weights[0][1]*tx3 - newron.bias/newron.weights[0][1]
    #ty4 = -1*newron.weights[0][0]/newron.weights[0][1]*tx3 - newron.bias/newron.weights[0][1]
    plt.plot(tx3, ty3, color='b', label='Threshold')
    #plt.plot(tx3, ty4, color='b', label='Threshold')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid()
    plt.title('Scatter Plot for threshold {}'.format(newron.threshold))
    fig.savefig(os.path.join(save_path, 'Q2P{}.png'.format(plotnum)), transparent = True)

nevaluate(newron, plotnum)
plotnum += 1

''' Part 4 '''

'''
thresh = np.arange(0, 1.2, 0.2)

for i in range(len(thresh)):
    print(plotnum)
    newron.train(learning_rate=0.1, threshold=thresh[i])
    nevaluate(newron, plotnum)
    plotnum += 1
'''

newron.train(learning_rate=0.1, threshold=0.5)
nevaluate(newron, plotnum)
plotnum += 1

newron.train(learning_rate=0.1, threshold=0.8)
nevaluate(newron, plotnum)
plotnum += 1

newron.train(learning_rate=0.1, threshold=1)
nevaluate(newron, plotnum)
plotnum += 1

newron.train(learning_rate=0.1, threshold=1.2)
nevaluate(newron, plotnum)
plotnum += 1

newron.train(learning_rate=0.1, threshold=1.5)
nevaluate(newron, plotnum)
plotnum += 1
