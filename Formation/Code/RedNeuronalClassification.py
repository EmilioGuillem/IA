from sklearn.model_selection import train_test_split
import tensorflow as tf
from  keras import layers, Sequential, optimizers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#Dataset
# X = np.random.randint(0, 100, (1000,2))/100
# y = np.sum(X, axis=1)

url = "https://gist.githubusercontent.com/jsz4n/b7ca11015784086788022a539935d0cf/raw/a8c3abf0a31f5c0df5e0ddd76fb9b289bac9bed1/winequality-red.csv"

df = pd.read_csv(url, sep=";")

print(df.head())
print(df.info())

X = df.copy()
y = X.pop("quality")
y=np.array(y).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#prettraitement
scaler = StandardScaler()
# scaler.fit(x_train)
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# ohe.fit(y_train)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(x_train_scaled.shape, y_train.shape)
print(x_test_scaled.shape, y_test.shape)

#MOdel
model =  Sequential()
model.add(layers.Input(shape=(11,), name="input"))
model.add(layers.Dense(30, activation="relu", name="layer1"))
model.add(layers.Dense(30, activation="relu", name="layer2"))
model.add(layers.Dense(6, activation="softmax", name="output"))

print(model.summary())


#compile
optimizer = optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy","precision"])

#Training
model.fit(x_train_scaled,y_train, epochs=30, validation_split=.2)

print(model.history.history)


#predict
y_pred = model.predict(x_test_scaled)
y_pred = np.array(y_pred).astype(np.int64).reshape(-1,1)


#evaluation
print(model.evaluate(x_test_scaled, y_test))

#post-processing
print(tf.argmax(y_pred,1))
t = ohe.get_feature_names_out()
print(t)

print(t[i] for i in tf.argmax(y_pred,1))


# print(ohe.inverse_transform(tf.argmax(y_pred,1)))



