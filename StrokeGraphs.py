# Creating 3d Graphs to display the relationships our model predicts and for comparison to the actual results

import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

plt.style.use('_mpl-gallery')


# Recreating model from the last program

model = tf.keras.Sequential([
      keras.layers.Dense(100, activation="sigmoid", input_dim=10), # tanh or elu works ok for just this layer as well, but sigmoid seems best
      keras.layers.Dense(100, activation="sigmoid", input_dim=100),
      keras.layers.Dense(100, activation="sigmoid", input_dim=100),
      keras.layers.Dense(1, activation="sigmoid", input_dim=100)
  ])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.load_weights('Models\Better Model')


# Recreating the test-data set to be used in the 3d scatter-plot graphs

data = pd.read_csv("brain_stroke.csv")
data['gender'] = data['gender'].map({'Male': 1, 'Female': 2})
data['ever_married'] = data['ever_married'].map({"Yes": 1, "No": 0})
data['Residence_type'] = data['Residence_type'].map({"Urban": 1, "Rural": 2})
data['work_type'] = data['work_type'].map({"children": 0, "Self-employed": 1, "Private": 2, "Govt_job": 3})
data['smoking_status'] = data["smoking_status"].map(
    {"never smoked": 1, "formerly smoked": 2, "smokes": 3, "Unknown": 0})

X = data.drop('stroke', axis=1).values
y = data['stroke'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=89)


# Scaling the Data

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)


# Predicts the stroke chances for the data

y_log = model.predict(X_test_scale)


# BMI, Age, Glucose Data Set Setup

size = len(y_test)
xA = [0] * size # Age (for x-axis)
yB = [0] * size # BMI (for y-axis)
yG = [0] * size # Avg Glucose Level (for y-axis)
for i in range(size):
        xA[i] = X_test[i, 1]
        yB[i] = X_test[i, 8]
        yG[i] = X_test[i, 7]

xB = yB # BMI (for x-axis)

zp = y_log # z-predicted: predicted stroke chance
zp *= 100
zt = y_test # z-test: actual stroke chance (occured or didn't occur)
zt *= 100


# Age vs BMI vs Stroke Graphs (xA, yB, zp + zt)

fig, ap1 = plt.subplots(subplot_kw={"projection": "3d"})
ap1.scatter(xA, yB, zp)
# ap1.set(xticklabels=[],yticklabels=[], zticklabels=[]) removes number labels
ap1.set_xlabel('Age')
ap1.set_ylabel('BMI')
ap1.set_zlabel('Predicted Stroke Percentage')
plt.title("Predicted Effect of Age and BMI on Stroke Chance\nRANGES FROM 0-30%")
plt.subplots_adjust(top=0.8)

fig, at1 = plt.subplots(subplot_kw={"projection": "3d"})
at1.scatter(xA, yB, zt)
at1.set_xlabel('Age')
at1.set_ylabel('BMI')
at1.set_zlabel('Actual Stroke Percentage')
plt.title("Actual Effect of Age and BMI on Stroke Chance\nRanges from 0 - 100% (no stroke & had a stroke)")
plt.subplots_adjust(top=0.8)


# Age vs Avg Glucose vs Stroke Graphs (xA, yG, zp + zt)

fig, ap2 = plt.subplots(subplot_kw={"projection": "3d"})
ap2.scatter(xA, yG, zp)
ap2.set_xlabel('Age')
ap2.set_ylabel('Avg Glucose Level(mg/dL)')
ap2.set_zlabel('Predicted Stroke Percentage')
plt.title("Predicted Effect of Age and Avg Glucose Level on Stroke Chance\nRANGES FROM 0-30%")
plt.subplots_adjust(top=0.8)

fig, at2 = plt.subplots(subplot_kw={"projection": "3d"})
at2.scatter(xA, yG, zt)
at2.set_xlabel('Age')
at2.set_ylabel('Avg Glucose Level (mg/dL)')
at2.set_zlabel('Actual Stroke Percentage')
plt.title("Actual Effect of Age and Avg Glucose Level on Stroke Chance\nRanges from 0 - 100% (no stroke & had a stroke)")
plt.subplots_adjust(top=0.8)


# BMI vs Avg Glucose vs Stroke Graphs (xB, yG, zp + zt)

fig, ap3 = plt.subplots(subplot_kw={"projection": "3d"})
ap3.scatter(xB, yG, zp)
ap3.set_xlabel('BMI')
ap3.set_ylabel('Avg Glucose Level (mg/dL)')
ap3.set_zlabel('Predicted Stroke Percentage')
plt.title("Predicted Effect of BMI and Avg Glucose Level on Stroke Chance\nRANGES FROM 0-30%")
plt.subplots_adjust(top=0.8)


fig, at3 = plt.subplots(subplot_kw={"projection": "3d"})
at3.scatter(xB, yG, zt)
at3.set_xlabel('BMI')
at3.set_ylabel('Avg Glucose Level(mg/dL)')
at3.set_zlabel('Actual Stroke Percentage')
plt.title("Actual Effect of BMI and Avg Glucose Level on Stroke Chance\nRanges from 0 - 100% (no stroke & had a stroke)")
plt.subplots_adjust(top=0.8)


# display
plt.show()
