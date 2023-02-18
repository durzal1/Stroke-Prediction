import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.style.use('_mpl-gallery')

model = keras.models.load_model('E:\PyCharm Community Edition 2022.2.4\Github\Stroke-Prediction\Models')

data = pd.read_csv("brain_stroke.csv")
data['gender'] = data['gender'].map({'Male': 1, 'Female': 2})
data['ever_married'] = data['ever_married'].map({"Yes": 1, "No": 0})
data['Residence_type'] = data['Residence_type'].map({"Urban": 1, "Rural": 2})
data['work_type'] = data['work_type'].map({"children": 0, "Self-employed": 1, "Private": 2, "Govt_job": 3})
data['smoking_status'] = data["smoking_status"].map(
    {"never smoked": 1, "formerly smoked": 2, "smokes": 3, "Unknown": 0})

X = data.drop('stroke', axis=1).values
y = data['stroke'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

y_log = model.predict(X_test)
# use X_test, y_test, y_log to create plot


# BMI, Age, Glucose Data Sets

xA = data['age'].values
yB = data['bmi'].vaues

xB = yB
yG = data['avg_glucose'].values

zp = y_log
zt = y_test

# Age vs BMI vs Stroke Chance (xA, yB, zp + zt)

fig, ap1 = plt.subplots(subplot_kw={"projection": "3d"})
ap1.scatter(xA, yB, zp)
ap1.set(xticklabels=[],
        yticklabels=[],
        zticklabels=[])
fig, at1 = plt.subplots(subplot_kw={"projection": "3d"})
at1.scatter(xA, yB, zt)
at1.set(xticklabels=[],
        yticklabels=[],
        zticklabels=[])

# Age vs Avg Glucose vs Stroke Chance (xA, yG, zp + zt)

fig, ap2 = plt.subplots(subplot_kw={"projection": "3d"})
ap2.scatter(xA, yG, zp)
ap2.set(xticklabels=[],
        yticklabels=[],
        zticklabels=[])
fig, at2 = plt.subplots(subplot_kw={"projection": "3d"})
at2.scatter(xA, yG, zt)
at2.set(xticklabels=[],
        yticklabels=[],
        zticklabels=[])

# BMI vs Avg Glucose vs Stroke Chance (xB, yG, zp + zt)

fig, ap3 = plt.subplots(subplot_kw={"projection": "3d"})
ap3.scatter(xB, yG, zp)
ap3.set(xticklabels=[],
        yticklabels=[],
        zticklabels=[])
fig, at3 = plt.subplots(subplot_kw={"projection": "3d"})
at3.scatter(xB, yG, zt)
at3.set(xticklabels=[],
        yticklabels=[],
        zticklabels=[])
