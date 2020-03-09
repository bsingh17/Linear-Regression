import numpy as np
import pandas as pd

dataset=pd.read_csv('energy vs efficiency.csv')
x=dataset.iloc[:,:-2].values
y=dataset.iloc[:,8:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

import seaborn as sns
sns.set_style('whitegrid') 
sns.lmplot(x ='X1', y ='Y1', data = dataset) 