import pandas as pd
from sklearn.base import accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

url = "https://gist.githubusercontent.com/jsz4n/b7ca11015784086788022a539935d0cf/raw/a8c3abf0a31f5c0df5e0ddd76fb9b289bac9bed1/titanic.csv"

df = pd.read_csv(url, index_col="PassengerId")

print(df.shape)
print(df.head())
print(df.info())

#preparation data
df["Cabin"].fillna("NoCabin")
categorical_features= ['Pclass', 'Sex', 'Embarked']
numerical_features= ['Age','Fare', 'Parch', 'SibSp']

to_remove = ["Name", "Ticket", "Cabin"]

df.drop(columns=to_remove, inplace=True)
df.dropna(inplace=True)

#Numerical features
scaler = MinMaxScaler() #numeros entre 0 et 1'
scaler.fit(df[numerical_features])
scaler.set_output(transform="pandas")

df[numerical_features] = scaler.transform(df[numerical_features])
df_numerical = pd.DataFrame(scaler.transform(df[numerical_features]), columns=numerical_features)

#categorical features
pd.get_dummies(df, columns=categorical_features)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.set_output(transform="pandas")
ohe.fit(df[categorical_features])
df_cat = pd.DataFrame(ohe.transform(df[categorical_features]))

df_final_X = pd.concat([df_cat, df_numerical], axis=1)



print(df_final_X)

X = df_final_X
Y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)


svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)

#Evalucation basique
eval = (y_pred == y_test).sum()/len(y_test) #valor cercano a 1
print(eval)

#Evaluation
(tn, fp) , (fn, tp) = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_true=y_test, y_pred=y_pred)) #cerano a 1
print(precision_score(y_true=y_test, y_pred=y_pred)) #cercano a 1
print(recall_score(y_true=y_test, y_pred=y_pred))#cercano a 1
print(f1_score(y_true=y_test, y_pred=y_pred))#cercano a 1



