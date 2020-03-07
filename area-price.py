import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv('e2.csv')

dataset.experience=dataset.experience.fillna(str('zero'))

import math
median_test_score=math.floor(dataset.test_score.median())
dataset.test_score=dataset.test_score.fillna(median_test_score)

from word2number import w2n
dataset.experience=dataset.experience.apply(w2n.word_to_num)

reg=LinearRegression()
reg.fit(dataset[['experience','test_score','interview_score']],dataset.salary)
print(reg.predict([[5,6,7]]))
