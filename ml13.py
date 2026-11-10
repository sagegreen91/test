import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("winequality-red.csv")

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for layers in [(50,), (100,), (50, 30), (100, 50, 25)]:
    model = MLPClassifier(hidden_layer_sizes=layers, max_iter=3000, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(layers, "Accuracy:", accuracy_score(y_test, pred))
