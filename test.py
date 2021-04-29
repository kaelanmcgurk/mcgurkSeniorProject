#%%
import pandas as pd
import altair as alt
import numpy as np
#%%

testTable = pd.read_csv('dataSets/leadsFinal.csv')

# %%
X = testTable.iloc[:, 1:9].values
y = testTable.iloc[:, 10].values
# %%
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics



ro = RandomOverSampler()

A = testTable.drop(['lead_id','status', 'isCust'],axis = 1)
b = testTable.filter(items = ['isCust'])

A_OS, b_OS = ro.fit_resample(A, b)

A_OS = pd.DataFrame(A_OS)
b_OS = pd.DataFrame(b_OS)


Atrain, Atest, bTrain, bTest = train_test_split(
    A_OS, 
    b_OS, 
    test_size=0.2
)


# evaluate random forest algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# define the model
model = RandomForestClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, A_OS, b_OS, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
#%%

# fit the model on the whole dataset
model.fit(Atrain, bTrain)
# make a single prediction
yhat = model.predict(Atest)
yProba = model.predict_proba(Atest)
# %%

#### MERGE THE PREDICTED DATA WITH THE ORIGINAL DATA