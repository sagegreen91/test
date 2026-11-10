import pandas as pd
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("mnist.csv")
X = df.drop('target', axis=1).values / 16.0
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

for activation in ['sigmoid', 'relu', 'tanh']:
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation=activation),
        Dense(64, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    start = time.time()
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    duration = time.time() - start
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\nActivation:", activation)
    print("Training Time:", round(duration, 2), "seconds")
    print("Accuracy:", accuracy_score(y_test, pred))
