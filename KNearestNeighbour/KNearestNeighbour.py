# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#def load_data(file_name):
#    data = pd.read_csv(file_name)
#    data = data.values
#    return data
#data = load_data('Outlier.txt')
#np.random.shuffle(data)


class KNearestNeighbour:
    
    def __init__(self,k = 5):
        self.k = k
    
    def calc_distance(self,x_train,x_test):
#        print(x_train)
#        print(x_test)
        d_sq_vector = np.square(x_train - x_test)
#        print(d_sq_vector)
        d_sq = np.sum(d_sq_vector)
#        print(d_sq)
        d = np.sqrt(d_sq)
#        print(d)       
        return d
    
    #def normalize(x):
    #    x_normal = (x-np.mean(x))/np.std(x)
    #    return x_normal
    
    def get_sort_arr(self,x_train,y_train,x_test):
        dist = np.zeros((x_train.shape[0],1))
        for i in range(x_train.shape[0]):
            dist[i] = self.calc_distance(x_train[i,:],x_test)
        dist_y = np.hstack((dist,y_train))
        dist_y = dist_y[dist_y[:,0].argsort()]
        return dist_y
            
    def predict(self,x,y,x_test):
        dist_y = self.get_sort_arr(x,y,x_test)
        k_nearest = dist_y[0:self.k,1:] 
        k_nearest = (k_nearest.flatten()).astype(int)
        k_count = np.bincount(k_nearest[:])
        prediction = np.argmax(k_count)
        return prediction
    
 
        