from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array ([[25,22],[35,25],[45,35],[50,40]])

y = np.array([0,0,1,1])

model = LogisticRegression()
model.fit(X,y)

result = model.predict([[40,34]])
print("Prediction:",result[0])