import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

#Load Dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns = iris.feature_names
)

y = iris.target 
print("Dataset Shape:", X.shape)
print("\nFirst 5 Records:")
print(X.head())

print("\nDataset Information:")
print(X.info())

print("\nStatistical Summary:")
print(X.describe())

plt.figure(figsize=(6,4))
sns.countplot(x=iris.target)
plt.title("Class Distribution")
plt.show()

# Pairplot
df = X.copy()
df["Species"] = y

sns.pairplot(df, hue="Species")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

 
# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Model Training

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)
recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)
f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

print("\nModel Performance")
print("-" * 30)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved as iris_model.pkl")


with open("iris_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully")




new_sample = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

prediction = loaded_model.predict(new_sample)

print("\nPrediction for New Sample:")
print("Class:", prediction[0])

print("Species:", iris.target_names[prediction[0]])