import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.datasets import fetch_california_housing


housing = fetch_california_housing() # this is a class. 
#to get to its attributes: print(dir(housing)) 
# dir stands for directory.it is used to examin attributes of modules, classes and objects.
print(dir(housing))

X = housing.data
Y = housing.target

#split the data into training and testing sets: random_size: seed for random number generation/shuffle: whether shuffling data before spliting
# strarify: ensures that the distribution of y variable is well maintained; in imbalanced data set (gender 5% girls and 95% boys) the spliting can get biased.
# parameres can get: None, array_like or list: Y: target variable  
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= .25, random_state=42, shuffle=True, stratify=None)

#create model
model = LinearRegression()

#train the model
model.fit(x_train, y_train)

#make prediction on the test set
y_pred = model.predict(x_test)

#evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'mean squared error: {mse}')


#plot the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()







