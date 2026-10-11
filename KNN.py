import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
data={
    "Study_Hours":[3,4,2,6,7,8],
    "Attended":[10,15,30,20,25,30],
    "Result":[0,0,0,1,1,1]
}
df=pd.DataFrame(data)

X=df[["Study_Hours","Attended"]]
Y=df["Result"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler=MinMaxScaler()
X_train_scaler=scaler.fit_transform(X_train)
print(X_train_scaler)
X_test_scaler=scaler.transform(X_test)
print(X_test_scaler)
print("\nMinimum Values Learned:")
print(scaler.data_min_)

print("\nMaximum Values Learned:")
print(scaler.data_max_)

model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaler,Y_train)
y_predict=model.predict(X_test_scaler)
print(y_predict)
newData={
    "Study_Hours":[2],
    "Attended":[10]
}
newDf=pd.DataFrame(newData)
new_X=newDf[["Study_Hours","Attended"]]
newData_scaler=scaler.transform(new_X)
newData_predict=model.predict(newData_scaler)
print(newData_predict)

