import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("breast-cancer.csv")
df = df.dropna()

X = df.drop('class', axis=1)
y = df['class']

# Convert categorical data into numbers
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'MLE': LogisticRegression(C=float('inf'), max_iter=5000),
    'MAP L1': LogisticRegression(C=1, l1_ratio=1, solver='saga', max_iter=5000),
    'MAP L2': LogisticRegression(C=1, l1_ratio=0, solver='saga', max_iter=5000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    print(name, "Accuracy:", accuracy_score(y_test, pred))
    print("Log Loss:", log_loss(y_test, prob))
    
