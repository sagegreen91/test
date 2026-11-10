import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("diabetes.csv")
df = df.dropna()
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {'Linear': make_pipeline(StandardScaler(), LinearRegression())}

ridge = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
    {'ridge__alpha': [0.01, 0.1, 1, 10, 100]}, cv=5)

lasso = GridSearchCV(make_pipeline(StandardScaler(), Lasso(max_iter=10000)),
    {'lasso__alpha': [0.01, 0.1, 1, 10]}, cv=5)
    
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

models['Ridge'] = ridge.best_estimator_
models['Lasso'] = lasso.best_estimator_

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "MSE:", mean_squared_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))

print("Best Ridge:", ridge.best_params_)
print("Best Lasso:", lasso.best_params_)
