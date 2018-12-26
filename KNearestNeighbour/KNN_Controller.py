import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from KNearestNeighbour import KNearestNeighbour

def load_data(file_name):
    data = pd.read_csv(file_name)
    data = data.values
    return data
data = load_data('Outlier.txt')
np.random.shuffle(data)
x_train = data[0:int(data.shape[0]*0.6) , 0:(data.shape[1]-1)]
y_train = data[0:int(data.shape[0]*0.6) , (data.shape[1]-1):]
x_test = data[int(data.shape[0]*0.6) : ,0:(data.shape[1]-1)]
y_test = data[int(data.shape[0]*0.6) : ,(data.shape[1]-1):]
def plot_func(x,y,x_label,y_label):
    for i in range(x.shape[0]):
        
        if y[i,:]==0:
            plt.scatter(x[i,0],x[i,1],color='red')
        elif y[i,:]==1:
            plt.scatter(x[i,0],x[i,1],color='blue')
        elif y[i,:]==2:
            plt.scatter(x[i,0],x[i,1],color='green')
        else:
            plt.scatter(x[i,0],x[i,1],color='yellow')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
plot_func(x_train,y_train,"Trained X","Trained Y")
plot_func(x_test,y_test,"Test X","Test Y" )
obj = KNearestNeighbour(k = 101)
correct = 0
wrong = 0
print ("Predicted    Y_Test")
for i in range(np.shape(x_test)[0]):   
    predicted = obj.predict(x_train,y_train,x_test[i])
    print (str(predicted) + "            " + str(y_test[i]))
    if(predicted == y_test[i]):
        correct += 1
    else:
        wrong +=1
accuracy = correct * 100 / (correct + wrong)
print("Accuracy = " + str(accuracy)+"%")




