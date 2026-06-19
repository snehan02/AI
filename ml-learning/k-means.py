from sklearn.cluster import KMeans
import numpy as np

X = np.array([
    [1000],
    [1200],
    [1500],
    [5000],
    [5500],
    [6000]
])

model = KMeans(
    n_clusters=2,
    random_state=42
)

model.fit(X)

print(model.labels_)