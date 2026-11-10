import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("20_newsgroup.csv")

vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = df['text'].fillna('')
X=vectorizer.fit_transform(X)

counts = np.asarray(X.sum(axis=0)).flatten()

mle = counts / counts.sum()
print("MLE:", mle[:10])

for alpha in [1.1, 2, 10]:
    map_estimate = (counts + alpha - 1) / (counts.sum() + len(counts) * (alpha - 1))
    print("Alpha:", alpha, "MAP:", map_estimate[:10])
