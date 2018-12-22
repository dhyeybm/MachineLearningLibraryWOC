# -*- coding: utf-8 -*-
import numpy as np


class NeuralNetworks():
    input_dimension = 0
    hidden_dimensions = 0
    Theta = None
    alpha = 0.3
    iterations = 1000
    
    def __init__(self,alpha = 0.5,iterations = 500):
        self.alpha = alpha
        self.iterations = iterations
    
    def normalize(self,x):
        x_normal = (x-np.mean(x))/np.std(x)
        return x_normal
          
    def get_params(self,Theta):
        Theta1 = np.reshape(Theta[0:np.prod(self.input_dimensions)],self.input_dimensions)    
        Theta2 = np.reshape(Theta[np.prod(self.input_dimensions):np.prod(self.hidden_dimensions)+np.prod(self.input_dimensions)],self.hidden_dimensions)
        return Theta1,Theta2
    
    def sigmoid(self,z):
        a = 1/(1 + np.exp(-1*z))
        return a
    
    def sigmoid_gradient(self,z):
        a =self.sigmoid(z)
        ans = a*(1-a)
        return ans
    
    #theta1 = layer x n+1 = 3 x 3
    #x_train = m x n = 599 x 2
    #theta2 = op x layer+1 = 4 x 4
    #y = m x op = 599 x 4
    def cost_func(self,x_train_norm,y_train,output_number,Theta):
        Theta1,Theta2 = self.get_params(Theta)
        m = x_train_norm.shape[0]
        a1 = x_train_norm
        a1 = np.hstack((np.ones((a1.shape[0],1)),a1)) #a1 = m x n+1 = 599 x 3
        z2 = np.dot(a1,np.transpose(Theta1))
        a2 = self.sigmoid(z2)            #m x layer = 3 x 599
        a2 = np.hstack((np.ones((a2.shape[0],1)),a2)) #m x layer+1 = 599 x 4
        z3 = np.dot(a2,np.transpose(Theta2)) #  m x op = 599 x 4
        h =  self.sigmoid(z3) 
        y = np.zeros((m,output_number))  
        for i in range(m):
            y[i,int(y_train[i,:])] = 1 
        del3 = h - y                # m x op
        del2 = (np.dot(del3,Theta2))*self.sigmoid_gradient(a2)  # m x layer+1
        del2 = del2[:,1: ]
        Theta1_grad = np.dot(np.transpose(del2),a1)/m    #layer+1 x n+1
        Theta2_grad = np.dot(np.transpose(del3),a2)/m   #  op x layer+1
        grad = Theta1_grad.flatten()
        grad = np.append(grad,Theta2_grad.flatten())  
        J = -1*sum(y_train * np.log(h) + (1-y_train) * np.log(1-h)) / m
        return J,grad,h
    
    def gradient_descent(self,x_train_norm,y_train,output_number,Theta):
        for i in range(self.iterations):
            J,grad,h = self.cost_func(x_train_norm,y_train,output_number,Theta)
            Theta = Theta - (self.alpha * grad)
            if i % 100 == 0:
                print ("Iterations : " + str(i) + "/" + str(self.iterations))
        return Theta
                  
    def neural_network(self,x_train,y_train):
        x_train_norm = self.normalize(x_train)
        Theta1 = np.random.rand(3,(np.shape(x_train_norm))[1]+1) * 0.01
        self.input_dimensions = np.shape(Theta1)
        output_number = len(np.unique(y_train))
        Theta2 = np.random.rand(output_number,4) * 0.01
        self.hidden_dimensions = np.shape(Theta2)
        self.Theta = Theta1.flatten()
        self.Theta = np.append(self.Theta,Theta2.flatten())
        self.Theta = self.gradient_descent(x_train_norm,y_train,output_number,self.Theta)
    
    def predict(self,x_test,y_test):
        
        x_test_norm = self.normalize(x_test)
        output_number = len(np.unique(y_test))  
        J,grad,h = self.cost_func(x_test_norm,y_test,output_number,self.Theta)
        m = x_test.shape[0]
        predict_matrix = np.zeros(m)
        for i in range(m):
            predict_matrix[i] = np.argmax(h,axis = 1)[i]
        return predict_matrix
    
    def accuracy(self,x_test,y_test):
        predicted = self.predict(x_test,y_test)
        correct = 0
        wrong = 0
        for i in range(x_test.shape[0]):
            if(y_test[i]==predicted[i]):
                correct+=1
            else:
                wrong+=1
        acc = (correct*100) / (correct+wrong)
        return acc
    
                
        
        
