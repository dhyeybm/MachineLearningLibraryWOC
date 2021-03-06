#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

class LinearRegression:
    
    theta = None
    y_mean = None
    y_std = None
    x_mean = None
    x_std = None
    m = None
    n = None
    alpha = None
    iterations = None
    
    def __init__(self,alpha = 0.1,iterations = 1000):
        self.alpha = alpha
        self.iterations = iterations
        
    def cost(self,x_norm,y_norm,h,theta):
        J_numerator = np.power(h-y_norm,2).sum()
        J = J_numerator / (2*self.m)
        grad = np.dot(np.transpose(x_norm),h-y_norm)
        return J,grad
    
    def descent(self,theta,alpha,iterations,x_norm,y_norm):
        for i in range(0,iterations):
            h = np.dot(x_norm,theta)
            J,grad = self.cost(x_norm,y_norm,h,theta)
            theta = theta - alpha * (grad / (self.m))
        return theta
    
    def calc_normalization_params(self, val):
        mean = np.mean(val, axis = 0)
        std_dev = np.std(val, axis = 0)
        return mean, std_dev
    
    def normalization(self, val, mean, std_dev):
        val = (val - mean) / std_dev
        return val
    
    def denormalize(self, val, mean, std_dev):
        return (val * std_dev) + mean
    
    def linear_regression(self, x_train, y_train): 
        self.x_mean, self.x_std = self.calc_normalization_params(x_train)
        self.y_mean, self.y_std = self.calc_normalization_params(y_train)
        x_norm = self.normalization(x_train, self.x_mean, self.x_std)
        y_norm = self.normalization(y_train, self.y_mean, self.y_std)
        self.m = x_norm.shape[0]
        self.n = x_norm.shape[1]
        ones = np.ones((self.m, 1),dtype = int)
        x_norm = np.hstack((ones, x_norm))
        theta = np.zeros(((self.n) + 1, 1),dtype = int)
        self.theta = self.descent(theta, self.alpha , self.iterations, x_norm, y_norm)
        
    def predict(self, x_test):
        x_test_norm = self.normalization(x_test, self.x_mean, self.x_std)
        m = x_test_norm.shape[0]
        ones = np.ones((m, 1), dtype = int)
        x_test_norm = np.hstack((ones, x_test_norm))
        y_pred = np.dot(x_test_norm, self.theta)
        y_pred_denormalized = self.denormalize(y_pred, self.y_mean, self.y_std)
        return y_pred_denormalized

    


 