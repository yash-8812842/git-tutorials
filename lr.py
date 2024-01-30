import pandas as pd
import numpy as np
from sklearn import *

data = pd.read_csv('Hosuing.csv')
data.head()

X = data.iloc[:,:-1]
y = data.iloc[:,-1]


scale = preprocessing.StandardScaler()
X = scale.fit_transform(X)

lr = linear_model.LinearRegression()
lr.fit(X,y)


