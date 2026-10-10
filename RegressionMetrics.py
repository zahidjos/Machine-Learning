import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
data={
    "Study_Hours":[3,4,5,6,7,8],
    "Attended":[10,15,30,20,25,30],
    "Result":[33.33,45,60,66.66,78.34,90]
}
df=pd.DataFrame(data)

X=df[["Study_Hours","Attended"]]
Y=df["Result"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42) 
model=LinearRegression()
model.fit(X_train,Y_train)
y_predict=model.predict(X_test)
print(y_predict)
mae=mean_absolute_error(Y_test,y_predict)
mse=mean_squared_error(Y_test,y_predict)
rmse=np.sqrt(mse)
r2=r2_score(Y_test,y_predict)
print(mae)
print(mse)
print(rmse)
print(r2)