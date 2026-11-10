import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("20_newsgroup.csv")
x=df['text']
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(X_train.fillna(''))
X_test = vectorizer.transform(X_test.fillna(''))

models = {
    'Multinomial NB': MultinomialNB(),
    'Bernoulli NB': BernoulliNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred, average='weighted'))
