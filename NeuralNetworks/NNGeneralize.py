#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class NNGeneralize:

    Theta = None
    
    def __init__(self,iterations = 1000,alpha = 0.03,labels = [5,2,3]):
        self.iterations = iterations
        self.alpha = alpha
        self.labels = labels
        self.n = len(self.labels) + 2
    
    def initialize_Theta(self):
        Theta = []
        for i in range(self.n-1):
            rand_matrix = (np.random.rand(self.labels[i+1],self.labels[i]+1)) 
            Theta.append(rand_matrix)
        return Theta
    
    def normalize(self,x):
        x_norm = (x - np.mean(x))/np.std(x)
        return x_norm
    
    def sigmoid(self,z):
        a = 1/(1+(np.exp(-1*z)))
        return a
    
    def sigmoid_gradient(self,a):
        sig_grad = a *(1-a)
        return sig_grad
   
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
        a[0] = np.hstack((np.ones((m,1)),a[0]))
        for i in range(1,self.n):
            z = np.dot(a[i-1],np.transpose(Theta[i-1]))
            a.append(self.sigmoid(z))
            a[i] = np.hstack((np.ones((m,1)),a[i]))
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
            delta[i] = (np.dot(delta[i+1],Theta[i]))*self.sigmoid_gradient(a[i])
            delta[i] = (delta[i])[:,1:]
        for i in range(self.n-1):
            gradient.append(np.dot(np.transpose(delta[i+1]),a[i])/m)
            grad_flat = np.reshape(gradient[i],(gradient[i].size,1))
            gradient_unrolled = np.append(gradient_unrolled,grad_flat)
            gradient_unrolled = gradient_unrolled
        return gradient_unrolled
    
    def gradient_descent(self,x_train_norm,y_train,theta_unrolled,iterations,alpha):
        for i in range(iterations):
            if(i%100 == 0):
                print("Iterations = " + str(i))
            gradient_unrolled = self.cost_func(x_train_norm,y_train,theta_unrolled)
            theta_unrolled = theta_unrolled - (self.alpha * gradient_unrolled)
        Theta = self.get_params(theta_unrolled)
        return Theta
    
    def max_freq(self,prob_matrix):
        predict = np.argmax(prob_matrix,axis = 1)
        return predict
            
    def neural_network(self,x_train,y_train):
       self.labels = [x_train.shape[1]] + self.labels + [len(np.unique(y_train))]
       self.Theta = self.initialize_Theta()
       x_train_norm = self.normalize(x_train)
       theta_unrolled = self.unroll(self.Theta)
       self.Theta = self.gradient_descent(x_train_norm,y_train,theta_unrolled,self.iterations,self.alpha)
       
    def prediction(self,x_test,y_test):
        x_test_norm = self.normalize(x_test)
        nn_matrix = self.forward_prop(x_test_norm,self.Theta) 
        prob_matrix = nn_matrix[self.n-1]
        prob_matrix = prob_matrix[:,1:]
        predict = self.max_freq(prob_matrix)
        return predict
    
    def accuracy(self,y_test,predict): 
        correct = 0
        wrong = 0
        for i in range(y_test.shape[0]):
            if y_test[i]==predict[i]:
                correct +=1
            else:
                wrong+=1
        return (correct*100/(correct+wrong))   
