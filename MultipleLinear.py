import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = {
    "Study_Hours":[3,4,5,6,7],
    "Attended":[10,15,30,20,25],
    "Result":[33.33,45,60,66.66,78.34]
}
df=pd.DataFrame(data)

x=df[["Study_Hours","Attended"]]
y=df["Result"]
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,Y_train)
y_predict=model.predict(X_test)
print(y_predict)
print(Y_test)
# r2=r2_score(Y_test,y_predict)
# print(r2)
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
new_student = [[8, 30],[7,15]]
new_pridict=model.predict(new_student)
print(new_pridict)
