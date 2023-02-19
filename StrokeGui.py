# User interface that predicts how likely they are to have a stroke using the ML Model and their personal data

import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tkinter as tk
from tkinter import *


# inits all the inputs

Sex = ""
Age = 0
HyperTension = ""
HeartDisease = ""
EverMarried = ""
WorkType = ""
Residence = ""
Glucose = 0
BMI = 0
EverSmoked = ""


# Function to calculate the chances of having a stroke from the provided biometrics
def getChance():

    # sets all the needed inputs
    Sex = str(sexVariable.get())
    Age = float(age.get())
    HyperTension = str(htvariable.get())
    HeartDisease = str(hdvariable.get())
    EverMarried = str(emvariable.get())
    WorkType = str(wtvariable.get())
    Residence = str(rtvariable.get())
    Glucose = float(avg_glucose.get())
    BMI = float(bmi.get())
    EverSmoked = str(ssvariable.get())

    # Maps the string values to integers
    if Sex == 'Male': Sex = 1
    else: Sex = 2
    if EverMarried == "Yes": EverMarried = 1
    else: EverMarried = 0
    if Residence == "Urban": Residence = 1
    else: Residence = 2
    if WorkType == "children": WorkType = 0
    elif WorkType == "Self-Employed": WorkType = 1
    elif WorkType == "Private": WorkType = 2
    else: WorkType = 3
    if EverSmoked == "never smoked": EverSmoked = 1
    elif EverSmoked == "formerly smoked": EverSmoked = 2
    else: EverSmoked = 3
    if HyperTension == "Yes": HyperTension = 1
    else: HyperTension = 0
    if HeartDisease == "Yes": HeartDisease = 1
    else: HeartDisease = 0

    # Destroys root

    root.destroy()


    # Loads up the model

    model = tf.keras.Sequential([
        keras.layers.Dense(100, activation="sigmoid", input_dim=10),
        keras.layers.Dense(100, activation="sigmoid", input_dim=100),
        keras.layers.Dense(100, activation="sigmoid", input_dim=100),
        keras.layers.Dense(1, activation="sigmoid", input_dim=100)
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


    # Loads in precomputed weights

    model.load_weights(r'Models\Better Model')


    # creates the test Data

    biometrics = [Sex, Age, HyperTension, HeartDisease, EverMarried, WorkType, Residence, Glucose, BMI, EverSmoked]
    bios = np.array(biometrics)

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

    for i in range(10):
        X_train[0, i] = bios[i]


    # Scales the data

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_test)


    # Tests the data

    chc = model.predict(X_train)
    chc = chc[0]


    str1 = ("PREDICTED CHANCE OF STROKE: " + str(chc*100) + "%")

    print(str1)

    # Uses tkinter to display the string

    root2 = Tk()
    root2.geometry("300x125")
    root2['bg'] = '#C5FEFF'

    chanceLabel = Label(root2, text=str1, bg='#87CEEB', fg='#022627')
    chanceLabel.place(anchor = 'center', relx = 0.5,
                   rely = 0.5,)

    tk.mainloop()


# Inits tkinter

root = Tk()
root.geometry("480x610")


# Button to Submit everything

go = Button(root, text="Enter", command=getChance, bg='#87CEEB', fg='#022627')
go.grid(row=10, column=0, columnspan=2, pady=25, padx=10, ipadx=140)


# Inits all the inputs
# These will all be float values that user can decide

age = Entry(root, width=30, bg='#87CEEB')
age.grid(row=1, column=1, padx=20, pady=(25, 0))

avg_glucose = Entry(root, width=30, bg='#87CEEB')
avg_glucose.grid(row=7, column=1, padx=20, pady=(25, 0))

bmi = Entry(root, width=30, bg='#87CEEB')
bmi.grid(row=8, column=1, padx=20, pady=(25, 0))


# Below are the drop-down values

# Sex
OPTIONS = [
"Male",
"Female",
]

sexVariable = StringVar(root)
sexVariable.set(OPTIONS[0]) # default value
it = sexVariable.get()
sex = OptionMenu(root, sexVariable, *OPTIONS)
sex.grid(row=0, column=1, padx=20, pady=(25, 0))
sex.config(bg='#87CEEB', fg='#022627')

# Hypertension
OPTIONS = [
"Yes",
"No",
]

htvariable = StringVar(root)
htvariable.set(OPTIONS[1]) # default value

hyperTension = OptionMenu(root, htvariable, *OPTIONS)
hyperTension.grid(row=2, column=1, padx=20, pady=(25, 0))
hyperTension.config(bg='#87CEEB', fg='#022627')


# Heart Disease
OPTIONS = [
"Yes",
"No",
]

hdvariable = StringVar(root)
hdvariable.set(OPTIONS[1]) # default value

heartDisease = OptionMenu(root, hdvariable, *OPTIONS)
heartDisease.grid(row=3, column=1, padx=20, pady=(25, 0))
heartDisease.config(bg='#87CEEB', fg='#022627')

# Ever Married
OPTIONS = [
"Yes",
"No",
]

emvariable = StringVar(root)
emvariable.set(OPTIONS[1]) # default value

everMarried = OptionMenu(root, emvariable, *OPTIONS)
everMarried.grid(row=4, column=1, padx=20, pady=(25, 0))
everMarried.config(bg='#87CEEB', fg='#022627')


# Work Type
OPTIONS = [
"Child",
"Self emplyed",
"Private",
"Government Job",
]

wtvariable = StringVar(root)
wtvariable.set(OPTIONS[0]) # default value

workType = OptionMenu(root, wtvariable, *OPTIONS)
workType.grid(row=5, column=1, padx=20, pady=(25, 0))
workType.config(bg='#87CEEB', fg='#022627')


# Residence Type
OPTIONS = [
"Urban",
"Rural",
]

rtvariable = StringVar(root)
rtvariable.set(OPTIONS[0]) # default value

residenceType = OptionMenu(root, rtvariable, *OPTIONS)
residenceType.grid(row=6, column=1, padx=20, pady=(25, 0))
residenceType.config(bg='#87CEEB', fg='#022627')


# Smoking Status
OPTIONS = [
"Never Smoked",
"Formerly Smoked",
"Smokes",
]

ssvariable = StringVar(root)
ssvariable.set(OPTIONS[0]) # default value

smokingStatus = OptionMenu(root, ssvariable, *OPTIONS)
smokingStatus.grid(row=9, column=1, padx=20, pady=(25, 0))
smokingStatus.config(bg='#87CEEB', fg='#022627')


# Plots all the inputs

sexLabel = Label(root, text="What is your biological sex?", bg='#87CEEB', fg='#022627')
sexLabel.grid(row=0, column=0, pady=(25, 0))

age_label = Label(root, text="What is your age?", bg='#87CEEB', fg='#022627')
age_label.grid(row=1, column=0, pady=(25, 0))

hyperTensionLabel = Label(root, text="Do you have Hyper Tension?", bg='#87CEEB', fg='#022627')
hyperTensionLabel.grid(row=2, column=0, pady=(25, 0))

heartDiseaseLabel = Label(root, text="Do you have Heart Disease?", bg='#87CEEB', fg='#022627')
heartDiseaseLabel.grid(row=3, column=0, pady=(25, 0))

everMarriedLabel = Label(root, text="Have you ever been married?", bg='#87CEEB', fg='#022627')
everMarriedLabel.grid(row=4, column=0, pady=(25, 0))

workTypeLabel = Label(root, text="What is your current state of work?", bg='#87CEEB', fg='#022627')
workTypeLabel.grid(row=5, column=0, pady=(25, 0))

residenceTypeLabel = Label(root, text="What is your current state of residence?", bg='#87CEEB', fg='#022627')
residenceTypeLabel.grid(row=6, column=0, pady=(25, 0))

avg_glucose_label = Label(root, text="What is your average glucose level(mg/dL)?", bg='#87CEEB', fg='#022627')
avg_glucose_label.grid(row=7, column=0, pady=(25, 0))

bmi_label = Label(root, text="What is your bmi?", bg='#87CEEB', fg='#022627')
bmi_label.grid(row=8, column=0, pady=(25, 0))

smokingStatusLabel = Label(root, text="What is your current smoking status?", bg='#87CEEB', fg='#022627')
smokingStatusLabel.grid(row=9, column=0, pady=(25, 0))

root['bg'] = '#C5FEFF'
# #87CEEB
tk.mainloop()

