import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

df = pd.read_csv("online_retail.csv", encoding="latin1")
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

df['Amount'] = df['Quantity'] * df['UnitPrice']

customer = df.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'Amount': 'sum',
    'InvoiceNo': 'nunique'
}).reset_index()

customer['Segment'] = pd.qcut(customer['Amount'], 3, labels=['Low', 'Medium', 'High'])

X = customer[['Quantity', 'InvoiceNo']]
y = customer['Segment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', max_depth=2)
model.fit(X_train, y_train)

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.show()

print("Feature Importance:", model.feature_importances_)
