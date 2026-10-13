
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_text
data = {
    "Monthly_Spend": [5000, 4500, 4000, 3500, 3000,
                      1000, 800, 1200, 1500, 2000],
    "Website_Visits": [12, 10, 9, 8, 7, 2, 1, 3, 4, 3],
    "Previous_Orders": [8, 6, 5, 4, 3, 1, 0, 1, 1, 2],
    "Bought_Again": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
}
df=pd.DataFrame(data)

X=df[["Monthly_Spend","Website_Visits","Previous_Orders"]]
Y=df["Bought_Again"]
model=DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)
model.fit(X,Y)
newData= pd.DataFrame({
    "Monthly_Spend":[500],
    "Website_Visits": [2],
    "Previous_Orders":[1]
})

new_predict=model.predict(newData)
print(new_predict)