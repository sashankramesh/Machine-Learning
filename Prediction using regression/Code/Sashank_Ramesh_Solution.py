# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 18:31:27 2020

@author: sasha
"""

"""
Question 1 : As a business like Property Guru, this use case could be extremely important. 
The features of the property are #mentioned in the dataset. Our data science model could 
offer insights to evaluate the feature that influences price of the property. Based on the budget 
and preferences of the property buyer, we can build a recommender system for the user that provides
the list of proerties for the user. 

Business statement: How does median income influence choice of home and hence house value?
Data science model applied will be a simple linear regression to predict the house value based on the median income?

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


input= pd.read_csv('C:\\Sashank\\Sashank\\SMU Fintech Courses\\Python exemption test\\housing\\anscombe.csv',sep=',')
input.columns
input.describe()

input2= pd.read_csv('C:\\Sashank\\Sashank\\SMU Fintech Courses\\Python exemption test\\housing\\housing.csv',sep=',')
input2.columns
summary=input2.describe()
input2.shape

#Question 2 :Data Preprocessing: We will consider input 2 as our dataset to preprocess

input2.isnull()
input2.isnull()
input3 = input2.dropna() #drop all null values
input3.shape #27 null values have been dropped

correlation =input3.corr() # Median income has a correlation of 0.68 with median house value

""" Question 3: Challenges while pre-processing the data: There is a need to impute the data due to lack of regression 
models to account for missing values. Since we are running a regression model, it is crucial to identify the variables 
that influence the outcome variable. In other cases, we must identify outlier data that might affect our data model"""

# Regression Model- Use Median_income of a neigbourhood to predict the median house value within the neighbourhood
input3.plot(x='median_income', y='median_house_value', style='o')  
plt.title('median_incomevs median_house_value')  
plt.xlabel('median_income')  
plt.ylabel('median_house_value')  
plt.show()

X = input3['median_income'].values.reshape(-1,1)
y = input3['median_house_value'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor= LinearRegression()
regressor.fit(X_train,y_train)

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test) # Predict value
Actvspred = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()}) #Actual vs predicted
plt.scatter(X_test, y_pred,  color='gray')
plt.show()



#Evaluating the performance of the model

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) # Mean absolute error
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) #Mean squared error
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #RootMean Squared error
	
"""
5) Potential problems while evaluating the model would be to identify the right metric by which we need to evaluate the
model. For instance, a simple linear regression model would need R squared value and the above mentioned metrics to
evaluate the model. In other cases such as classification problems, we would need to identify the ROC curves and identify 
the best model amongst them. In this dataset, the mean absolute error between actual and predicted values are actually high
and hence, the model isn't necessarily effective to predit the median house value. 

"""
