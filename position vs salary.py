import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('position vs salary.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=8)
x_poly=poly.fit_transform(x)
poly.fit(x_poly,y)
reg2=LinearRegression()
reg2.fit(x_poly,y)

plt.title('Polynomial Regression')
plt.scatter(x,y)
plt.xlabel('Level')
plt.ylabel('Salary')
plt.plot(x,reg.predict(x),color='red')
plt.plot(x,reg2.predict(x_poly),color='green')