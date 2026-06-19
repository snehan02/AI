from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()

X = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(X,y)

result = model.predict([[5.1,3.5,1.4,0.2]])

print("Species:",iris.target_names[result[0]])