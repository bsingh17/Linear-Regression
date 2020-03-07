import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv('e4.csv')

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]

plt.scatter(x,y,color='red')
reg=LinearRegression()
reg.fit(x,y)
plt.plot(x,reg.predict(x))
print(reg.predict([[2020]]))