from sklearn.linear_model import LogisticRegression
X = [[1], [2], [3], [4], [5], [6]]
Y=[0,0,0,1,1,1]
model=LogisticRegression()
model.fit(X,Y)
result=model.predict([[8]])
print(result[0])
result_proba=model.predict_proba([[10]])
print(result_proba[0][1])
if result_proba[0][1]>=0.5:
    print("pass")
else:
    print("fail")
