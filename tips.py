import numpy as np
import pandas as pd

dataset=pd.read_csv('tips.csv')

from sklearn.preprocessing import LabelEncoder
lbl_sex=LabelEncoder()
dataset['sex']=lbl_sex.fit_transform(dataset['sex'])
lbl_smoker=LabelEncoder()
dataset['smoker']=lbl_smoker.fit_transform(dataset['smoker'])
lbl_day=LabelEncoder()
dataset['day']=lbl_day.fit_transform(dataset['day'])
lbl_time=LabelEncoder()
dataset['time']=lbl_time.fit_transform(dataset['time'])
x=dataset.iloc[:,1:].values
y=dataset.iloc[:,0:1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

print(reg.score(x_test,y_test))