from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([
    [2],
    [3],
    [5],
    [7],
    [8]
])

y = np.array([
    0,
    0,
    1,
    1,
    1
])

model = DecisionTreeClassifier()
model.fit(X,y)

result = model.predict([[4]])

print("Result:",result[0])