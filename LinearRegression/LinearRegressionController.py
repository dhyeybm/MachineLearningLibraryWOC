#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

    
def load_data(file_name):
    train_data = pd.read_csv(file_name)
    train_data = train_data.values
    return train_data

train_data = load_data('LinearRegressionData.txt')
dimensions = train_data.shape
x_train = np.array(train_data[0:40, 0:dimensions[1]-1])
y_train = np.array(train_data[0:40, dimensions[1]-1:])
x_test = np.array(train_data[0:6, 0:dimensions[1]-1])
y_test = np.array(train_data[0:6, dimensions[1]-1:])
regr = LinearRegression()
regr.linear_regression(x_train, y_train)
y_pred = regr.predict(x_test)
plt.scatter(x_train[:, 0], y_train, color = 'Red')
plt.scatter(x_test[:, 0], y_test, color = 'Blue')
plt.plot(x_test[:, 0], y_pred)
plt.show()