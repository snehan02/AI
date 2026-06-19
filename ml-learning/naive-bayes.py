from sklearn.naive_bayes import GaussianNB
import numpy as np

X = np.array([
    [1],
    [2],
    [8],
    [9]
])

y = np.array([
    0,
    0,
    1,
    1
])

model = GaussianNB()
model.fit(X,y)

result = model.predict([[7]])

print(result[0])