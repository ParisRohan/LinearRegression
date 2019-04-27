#import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("hours.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
print("Accuracy :", regressor.score(X,y) * 100)

hours=int(input('enter the no of hours:')) 

eq=regressor.coef_*hours+regressor.intercept_ #to show formula
print('y = %f*%f+%f ' %(regressor.coef_,hours,regressor.intercept_))
print("y=" , eq[0])

plt.scatter(X,y,color='red') #to point data entries on graph 
plt.plot(X,regressor.predict(X),color='blue') #to plot regression line
plt.title("Linear Regression")
plt.xlabel("Independent variables")
plt.ylabel("Dependent variables")
plt.show()
