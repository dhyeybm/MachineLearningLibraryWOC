#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    data = pd.read_csv(file_name,sep = ',')
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

class NNGeneralize:

    Theta = None
    n = 0
    
    def __init__(self,iterations = 1000,alpha = 0.3,labels = [5,2,3]):
        self.iterations = iterations
        self.alpha = alpha
        self.labels = labels
    
    def initialize_Theta(self):
        Theta = []
        for i in range(self.n-1):
            rand_matrix = np.random.rand(self.labels[i+1],self.labels[i]+1)
            Theta.append(rand_matrix)   
        return Theta
    
    def normalize(self,x):
        x_norm = (x - np.mean(x))/np.std(x)
        return x_norm
    
    def sigmoid(self,z):
        a = 1/(1+(np.exp(-1*z)))
        return a
    
    
    def unroll(self,Theta):
        theta_unrolled = np.array([])
        for i in range(self.n-1):
            theta_unrolled = np.append(theta_unrolled,Theta[i].flatten())
        return theta_unrolled
    
    def get_params(self,theta_unrolled):
        Theta = []
        Theta.append(np.reshape(theta_unrolled[0:self.labels[1]*(self.labels[0]+1)],(self.labels[1],self.labels[0]+1)))
        first_index = (self.labels[0]+1) * self.labels[1]
        for i in range(1,self.n-2):
            last_index = first_index + (self.labels[i]+1) * self.labels[i+1]
            Theta_matrix = np.reshape(theta_unrolled[first_index : last_index],(self.labels[i+1],self.labels[i]+1))
            Theta.append(Theta_matrix)
            first_index = last_index
        last_matrix = np.reshape(theta_unrolled[(theta_unrolled.size - (self.labels[self.n-2]+1)*self.labels[self.n-1]):] ,((self.labels[self.n-1]),self.labels[self.n-2]+1))
        Theta.append(last_matrix)
        return Theta
    
    def forward_prop(self,x_train_norm,Theta):
        a = [x_train_norm]
        m = x_train_norm.shape[0]
        print(self.labels)
        print(Theta)
        a[0] = np.hstack((np.ones((m,1)),a[0]))
        for i in range(1,2):
            print(np.dot(a[i-1],np.transpose(Theta[i-1])))
            a.append(self.sigmoid(np.dot(a[i-1],np.transpose(Theta[i-1]))))
            print(a[1])
            a[i] = np.hstack((np.ones((m,1)),a[i]))
            #print(a)
        return a
    
    
    def cost_func(self,x_train_norm,y_train,theta_unrolled):
        Theta = self.get_params(theta_unrolled)
        m = x_train_norm.shape[0]
        gradient = []
        gradient_unrolled = np.array([])
        delta = [0 for i in range(self.n)]
        a = self.forward_prop(x_train_norm,Theta)
        y_matrix = np.zeros((m,self.labels[self.n-1]))
        for i in range(m):
            y_matrix[i,int(y_train[i])] = 1
        delta[self.n-1] = (a[self.n-1])[:,1:] - y_matrix 
        for i in range(self.n-2,0,-1):
            delta[i] = np.dot(delta[i+1],Theta[i])
            delta[i] = (delta[i])[:,1:]
        for i in range(self.n-1):
            gradient.append(np.dot(np.transpose(delta[i+1]),a[i])/m)
            grad_flat = np.reshape(gradient[i],(gradient[i].size,1))
            gradient_unrolled = np.append(gradient_unrolled,grad_flat)
            gradient_unrolled = gradient_unrolled/m
        return gradient_unrolled
    
    def gradient_descent(self,x_train_norm,theta_unrolled,iterations,alpha):
        for i in range(iterations):
            gradient_unrolled = self.cost_func(x_train_norm,y_train,theta_unrolled)
            theta_unrolled = theta_unrolled - (self.alpha * gradient_unrolled)
            #print(str(theta_unrolled[9]) + "         " + str(gradient_unrolled[9]))
        Theta = self.get_params(theta_unrolled)
        #print(Theta)
        return Theta
    
    def max_freq(self,prob_matrix):
        predict = np.argmax(prob_matrix,axis = 1)
        return predict
                
    def neural_network(self,x_train,y_train):
       self.labels = [x_train.shape[1]] + self.labels + [len(np.unique(y_train))]
       self.n = len(self.labels)
       self.Theta = self.initialize_Theta()
       x_train_norm = self.normalize(x_train)
       theta_unrolled = self.unroll(self.Theta)
       self.Theta = self.gradient_descent(x_train_norm,theta_unrolled,self.iterations,self.alpha)
       #print(self.Theta)
       
    def prediction(self,x_test,y_test):
        x_test_norm = self.normalize(x_test)
        #print(self.Theta)
        nn_matrix = self.forward_prop(x_test_norm,self.Theta) 
        prob_matrix = nn_matrix[self.n-1]
        prob_matrix = prob_matrix[:,1:]
        predict = self.max_freq(prob_matrix)
        #print(prob_matrix)
        return predict
    


obj = NNGeneralize(alpha = 0.5)
obj.neural_network(x_train,y_train)
predict = obj.prediction(x_test,y_test)
print()
print(predict)
correct = 0.0
wrong = 0.0
for i in range(x_test.shape[0]):

    if y_test[i]==predict[i]:
        correct +=1
    else:
        wrong+=1
print(correct*100/(correct+wrong))