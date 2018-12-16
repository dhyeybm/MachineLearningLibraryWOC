# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
    alpha = 0.1
    iterations = 1000
    theta = None
    m = 0
    n = 0 
    def sigmoid(self,theta,x_train):
        z = np.dot(x_train,theta)
        h = 1/(1 + np.exp(-1 * z))
        return h
        
    def cost(self,x_train_norm,y_train,theta):
        h = self.sigmoid(theta,x_train_norm)
        J_num = (y_train * np.log(h)) + ((1 - y_train) * np.log(1 - h))
        J = -(J_num)/self.m
        xt=np.transpose(x_train_norm)
        grad_num = np.dot(xt, h-y_train)
        grad = grad_num/self.m
        return J,grad
    
    def gradient_descent(self,alpha,iterations,theta,x_train_norm,y_train):    
        for i in range(0, iterations):
            J,grad = self.cost(x_train_norm,y_train,theta)            
            theta = theta - alpha * grad
        return theta
        
    def normal(self,val):
        val_norm = (val - np.mean(val)) / (np.std(val))
        return val_norm
        
    def approx(self,h):
        temp = np.zeros((h.shape[0],1))
        for i in range(h.shape[0]):
            if(h[i,0] >= 0.5):
                temp[i,0] = 1
            else:
                temp[i,0] = 0
        return temp
    
    def logistic_regression(self,x_train,y_train):
        self.m = x_train.shape[0]
        self.n = x_train.shape[1]
        x_train_norm = self.normal(x_train)
        self.theta = np.zeros((self.n,1))
        self.theta = self.gradient_descent(self.alpha,self.iterations,self.theta,x_train_norm,y_train)
    
    def predict_probability(self,x_test):
        x_test_norm = self.normal(x_test)
        y_prob = self.sigmoid(self.theta,x_test_norm)
        return y_prob
    
    def predict_y(self,x_test):
        y_pred = self.predict_probability(x_test)
        return self.approx(y_pred)
    
    def plot_train(self,x_train,y_train):
        for i in range(x_train.shape[0]):
            if y_train[i] == 0:
                plt.scatter(x_train[i,0],x_train[i,1],marker='_',color='red')
            else :
                plt.scatter(x_train[i,0],x_train[i,1],marker='+',color='blue')
        plt.xlabel('Trained X')
        plt.ylabel('Trained Y')
        plt.show()    
        
    def plot_test(self,x_test,y_test):
        for i in range(x_test.shape[0]):
            if y_test[i] == 0:
                plt.scatter(x_test[i,0],x_test[i,1],marker='_',color='red')
            else :
                plt.scatter(x_test[i,0],x_test[i,1],marker='+',color='blue')
        plt.xlabel('Test X')
        plt.ylabel('Test Y')
        plt.show() 
            
    def accuracy(self,x_test,y_test):
        y_pred = self.predict_y(x_test)
        correct = 0.0
        wrong = 0.0 
        for i in range(y_test.shape[0]):
            if(y_test[i] == y_pred[i]):
                correct += 1
            else :
                wrong += 1
        acc = correct / (correct + wrong)
        return acc
    
            
        
               

      


        

        