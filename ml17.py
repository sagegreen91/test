import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df=pd.read_csv("digits.csv")
x=df.drop("label",axis=1)
x=StandardScaler().fit_transform(x)

inertias,scores=[],[]

for k in [5,10,15,20]:
    print("processing k=",k,flush=True)
    model=KMeans(n_clusters=k,random_state=42,max_iter=100,n_init=5)
    labels=model.fit_predict(x)
    inertias.append(model.inertia_)
    score=silhouette_score(x,labels,sample_size=100,random_state=42)
    scores.append(score)
    print("k:",k,"inertia",model.inertia_,"silhouette",score,flush=True)
    
plt.plot([5,10,15,20],inertias,marker='o')
plt.xlabel("k")
plt.ylabel("inertias")
plt.show()

plt.plot([5, 10, 15, 20], scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()
