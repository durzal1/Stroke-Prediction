import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBRFClassifier

data = pd.read_csv("brain_stroke.csv")

# data.head(1) ?? not sure what it does

data.info() # gives a small summary view

# deleting all the values that are not numerical
#TODO add this back

# data['gender'].value_counts()
# data = pd.get_dummies(data, columns=['gender'], drop_first=True) # removes gender column

data['gender'] = data['gender'].map({'Male':1, 'Female':2})
data['ever_married'] = data['ever_married'].map({"Yes":1, "No":0})
data['Residence_type'] = data['Residence_type'].map({"Urban":1, "Rural":2})
data['work_type'] = data['work_type'].map({"children": 0, "Self-employed":1, "Private":2, "Govt_job": 3})
data['smoking_status'] = data["smoking_status"].map({"never smoked":1, "formerly smoked":2, "smokes":3, "Unknown": 0})

data.isnull().values.any()

X = data.drop('stroke',axis=1).values
y = data['stroke'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



rfc = RandomForestClassifier(n_estimators=100) # the number of trees in the forest
rfc.fit(X_train, y_train)
