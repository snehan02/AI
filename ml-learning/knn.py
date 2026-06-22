from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X = np.array([
    [100],
    [120],
    [200],
    [220]
])

y = np.array([
    "Apple",
    "Apple",
    "Orange",
    "Orange"
])

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X,y)

result = model.predict([[110]])

print(result[0])