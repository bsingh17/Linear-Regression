import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train=pd.read_csv('train.csv')
dataset_test=pd.read_csv('test.csv')

x_train=dataset_train.iloc[:,:-1].values
y_train=dataset_train.iloc[:,1].values
x_test=dataset_test.iloc[:,:-1].values
y_test=dataset_test.iloc[:,1].values

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,reg.predict(x_test))