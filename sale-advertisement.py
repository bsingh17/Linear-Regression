import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('sales-advertisement.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x,y)
y_predict=reg.predict(x)

plt.xlabel('Sales')
plt.ylabel('advertisment cost')
plt.scatter(x,y,color='red')
plt.plot(x,reg.predict(x))

print(reg.coef_)
print(reg.intercept_)