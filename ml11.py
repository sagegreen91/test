import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("iris-data.csv")

X = df[['sepal length', 'sepal width']]
y = (df['class'] == 'Iris-setosa').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

w = model.coef_[0]
b = model.intercept_[0]

x = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 100)

boundary = -(w[0] * x + b) / w[1]
margin1 = -(w[0] * x + b - 1) / w[1]
margin2 = -(w[0] * x + b + 1) / w[1]

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
plt.plot(x, boundary, 'k-')
plt.plot(x, margin1, 'r--')
plt.plot(x, margin2, 'r--')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM - Iris Classification")
plt.show()
