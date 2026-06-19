from sklearn.svm import SVC
import numpy as np

X = np.array([
    [1],
    [2],
    [3],
    [8],
    [9],
    [10]
])

y = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

model = SVC()

model.fit(X,y)

result = model.predict([[7]])

print(result[0])