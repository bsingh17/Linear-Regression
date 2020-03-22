import pandas as pd
import numpy as np

dataset=pd.read_csv('school_grades_dataset.csv')

from sklearn.preprocessing import LabelEncoder
lbl_school=LabelEncoder()
dataset['school']=lbl_school.fit_transform(dataset['school'])
lbl_sex=l=LabelEncoder()
dataset['sex']=lbl_sex.fit_transform(dataset['sex'])
lbl_address=LabelEncoder()
dataset['address']=lbl_address.fit_transform(dataset['address'])
lbl_famsize=LabelEncoder()
dataset['famsize']=lbl_famsize.fit_transform(dataset['famsize'])
lbl_pstatus=LabelEncoder()
dataset['Pstatus']=lbl_pstatus.fit_transform(dataset['Pstatus'])
lbl_mjob=LabelEncoder()
dataset['Mjob']=lbl_mjob.fit_transform(dataset['Mjob'])
lbl_fjob=LabelEncoder()
dataset['Fjob']=lbl_fjob.fit_transform(dataset['Fjob'])
lbl_reason=LabelEncoder()
dataset['reason']=lbl_reason.fit_transform(dataset['reason'])
lbl_guardian=LabelEncoder()
dataset['guardian']=lbl_guardian.fit_transform(dataset['guardian'])
lbl_schoolsup=LabelEncoder()
dataset['schoolsup']=lbl_schoolsup.fit_transform(dataset['schoolsup'])
lbl_famsup=LabelEncoder()
dataset['famsup']=lbl_famsup.fit_transform(dataset['famsup'])
lbl_paid=LabelEncoder()
dataset['paid']=lbl_paid.fit_transform(dataset['paid'])
lbl_activities=LabelEncoder()
dataset['activities']=lbl_activities.fit_transform(dataset['activities'])
lbl_nursery=LabelEncoder()
dataset['nursery']=lbl_nursery.fit_transform(dataset['nursery'])
lbl_higher=LabelEncoder()
dataset['higher']=lbl_higher.fit_transform(dataset['higher'])
lbl_internet=LabelEncoder()
dataset['internet']=lbl_internet.fit_transform(dataset['internet'])
lbl_romantic=LabelEncoder()
dataset['romantic']=lbl_romantic.fit_transform(dataset['romantic'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

print(reg.score(x_test,y_test))