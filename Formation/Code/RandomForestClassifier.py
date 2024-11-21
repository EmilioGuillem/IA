#usar DecisionTreeClassifier() ya que no necesita tanto y es más preciso


from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier


stc_dt = SelfTrainingClassifier(DecisionTreeClassifier)
stc_dt.fit(x_train, y_train)
y_train_stc = stc_dt.predict(x_test)


# use pipeline
# pipelinepreporcessing
# pipeline = Pipeline(steps[])
