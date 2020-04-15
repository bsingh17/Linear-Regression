import numpy as np
import pandas as pd

dataset=pd.read_csv('MARUTI.csv')
dataset=dataset.drop(['Symbol','Series','Date'],axis='columns')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
x=scaler.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=100)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

print(model.score(x_test,y_test))
