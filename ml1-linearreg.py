import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("california_housing.csv")
df = df.dropna(subset=['total_rooms', 'median_house_value'])

x=df['total_rooms'].values
y=df['median_house_value'].values

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

xm,xs=xtrain.mean(),xtrain.std()
ym,ys=ytrain.mean(),ytrain.std()

xtrain=(xtrain-xm)/xs
xtest=(xtest-xm)/xs
ytrain_scaled=(ytrain-ym)/ys

xtrain=np.c_[np.ones(len(xtrain)),xtrain]
xtest=np.c_[np.ones(len(xtest)),xtest]

theta=np.zeros(2)
lr=0.01

for i in range(1000):
    error=xtrain@theta-ytrain_scaled
    theta-=lr*(2/len(ytrain))*xtrain.T@error

ypred=(xtest@theta)*ys+ym

print("gradient descent")
print("mse:",mean_squared_error(ytest,ypred))
print("r2 score:",r2_score(ytest,ypred))

theta=np.linalg.pinv(xtrain)@ytrain
ypred2=xtest@theta

print("normal eqn")
print("mse:",mean_squared_error(ytest,ypred2))
print("r2 score:",r2_score(ytest,ypred2))

idx=np.argsort(xtest[:,1])
plt.scatter(xtest[:,1],ytest)
plt.plot(xtest[idx,1],ypred2[idx],'r')
plt.xlabel("tot rooms")
plt.ylabel("med house val")
plt.show()
