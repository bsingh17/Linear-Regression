import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('salary.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
simplelinearregression=LinearRegression()
simplelinearregression.fit(x_train,y_train)
y_predict=simplelinearregression.predict(x_test)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,simplelinearregression.predict(x_train))
plt.show()
