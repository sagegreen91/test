import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("auto-mpg.csv").dropna()

X = df[['displacement']]
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

for degree in [1, 2, 3, 4]:
    poly = PolynomialFeatures(degree)
    Xtr = poly.fit_transform(X_train)
    Xte = poly.transform(X_test)

    model = LinearRegression().fit(Xtr, y_train)
    pred = model.predict(Xte)

    print("Degree:", degree)
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))

    xp = pd.DataFrame({'displacement': np.linspace(X.min().iloc[0], X.max().iloc[0], 100)})
    yp = model.predict(poly.transform(xp))

    plt.scatter(X, y, alpha=0.3)
    plt.plot(xp, yp, label="Degree " + str(degree))

plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.legend()
plt.show()
