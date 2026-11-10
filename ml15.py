import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

df = pd.read_csv("fashion-mnist_test.csv")

X = df.drop('label', axis=1).values / 255.0
y = df['label']

print("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

best_acc = 0

for lr in [0.001, 0.01]:
    for batch in [32, 64]:
        for ep in [5, 10]:
            model = Sequential([
                Input(shape=(X_train.shape[1],)),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(10, activation='softmax')
            ])

            model.compile(
                optimizer=Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

            model.fit(X_train, y_train, epochs=ep,batch_size=batch, verbose=0)

            loss, acc = model.evaluate(X_test, y_test, verbose=0)

            print("LR:", lr, "Batch:", batch,"Epochs:", ep, "Accuracy:", round(acc, 4))

            if acc > best_acc:
                best_acc = acc
                best_params = (lr, batch, ep)

print("\nBest Parameters:", best_params)
print("Best Accuracy:", round(best_acc, 4))
