import numpy as np
import pandas as pd

dataset_train=pd.read_csv('train.csv')
dataset_train=dataset_train.drop(['id','pickup_datetime','dropoff_datetime'],axis='columns')
dataset_test=pd.read_csv('test.csv')
dataset_test=dataset_test.drop(['id','pickup_datetime'],axis='columns')

from sklearn.preprocessing import LabelEncoder
lbl_store_train=LabelEncoder()
dataset_train['store_and_fwd_flag']=lbl_store_train.fit_transform(dataset_train['store_and_fwd_flag'])
lbl_store_test=LabelEncoder()
dataset_test['store_and_fwd_flag']=lbl_store_test.fit_transform(dataset_test['store_and_fwd_flag'])

x_train=dataset_train.iloc[:,:-1].values
y_train=dataset_train.iloc[:,-1].values
x_test=dataset_test.iloc[:,:].values

import math
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test).floor

