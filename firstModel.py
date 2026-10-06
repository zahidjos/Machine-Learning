import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import(mean_absolute_error,mean_squared_error,r2_score)

data={
    "Study Hours":[2,3,4,5,6,7,8,9],
    "Marks":[20,30,40,50,60,70,80,90]
}
df=pd.DataFrame(data)
x=df[["Study Hours"]]
y=df["Marks"]

X_train,X_test,Y_train,Y_test=train_test_split(x, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train,Y_train)
y_predict=model.predict(X_test)

mae=mean_absolute_error(Y_test,y_predict)
mse=mean_squared_error(Y_test,y_predict)
rmse=np.sqrt(mse)
r2=r2_score(Y_test,y_predict)
print(mae)
print(mse)
print(rmse)
print(r2)
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# 9. Predict new data
new_data = pd.DataFrame({
    "Study Hours": [200]
})

prediction = model.predict(new_data)

print("Predicted Marks:", prediction[0])


