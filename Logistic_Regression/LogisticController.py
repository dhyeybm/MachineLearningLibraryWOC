#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
from LogisticRegression import LogisticRegression


filename = 'LogisticRegressionData1.txt'
def load_data(filename):
    data = pd.read_csv(filename)
    data=data.values
    return data


data = load_data(filename)
dimensions = data.shape
x_train = data[0:50,0:dimensions[1]-1]
y_train = data[0:50,dimensions[1]-1:]
m = x_train.shape[0]
n = x_train.shape[1] 
x_test = data[51:,0:dimensions[1]-1]
y_test = data[51:,dimensions[1]-1:]
tester = LogisticRegression()
tester.logistic_regression(x_train,y_train)
pred_probability = tester.predict_probability(x_test)
print "Probability of y being 1 for given testing samples:" + str(pred_probability)
pred_y = tester.predict_y(x_test)
print "Predicted value of y for given testing samples : " + str(pred_y)
plot_train = tester.plot_train(x_train,y_train)
plot_test = tester.plot_test(x_test,y_test)
acc = tester.accuracy(x_test,y_test)
print "Accuracy of regression algorithm = " + str(acc)
                             



