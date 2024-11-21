import tensorflow as tf
from  keras import layers, Sequential
import numpy as np

#Dataset
X = np.random.randint(0, 100, (1000,2))/100
y = np.sum(X, axis=1)

print(X.shape,y.shape)


#MOdel
model =  Sequential()
model.add(layers.Input(shape=(2,), name="input"))
model.add(layers.Dense(10, activation="relu", name="layer1"))
model.add(layers.Dense(4, activation="linear", name="layer2"))
model.add(layers.Dense(1, name="output"))

print(model.summary())


#compile
model.compile(optimizer="SGD", loss="MSE", metrics=["accuracy"])

#Training
model.fit(X,y, epochs=20)

print(model.history.history)


#predict
print((model.predict(X)*100 - y*100).sum()/1000) #get error

#evaluation

