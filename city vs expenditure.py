import numpy as np
import pandas as pd

dataset=pd.read_csv('e3.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lbl=LabelEncoder()
x[:,3]=lbl.fit_transform(x[:,3])
ohd=OneHotEncoder(categorical_features=[3])
x=ohd.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train , y_train)
y_predict=reg.predict(x_test)

print(reg.intercept_)
print(reg.coef_)
