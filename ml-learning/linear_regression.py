from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1000],[1500],[2000],[2500]])
y = np.array([20,30,40,50])

model = LinearRegression()
model.fit(X,y)

price = model.predict([[3000]])
print("Predicted Price:",price[0])