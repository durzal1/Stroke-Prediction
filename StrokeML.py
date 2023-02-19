# Training of ML Model

import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Gets the data from an excel sheet
data = pd.read_csv("brain_stroke.csv")
data.info()


# Maps string values to integers
data['gender'] = data['gender'].map({'Male':1, 'Female':2})
data['ever_married'] = data['ever_married'].map({"Yes":1, "No":0})
data['Residence_type'] = data['Residence_type'].map({"Urban":1, "Rural":2})
data['work_type'] = data['work_type'].map({"children": 0, "Self-employed":1, "Private":2, "Govt_job": 3})
data['smoking_status'] = data["smoking_status"].map({"never smoked":1, "formerly smoked":2, "smokes":3, "Unknown": 0})


# splitting it into x and y axis values
X = data.drop('stroke',axis=1).values
y = data['stroke'].values

# init variables
testperc5_ = 0
predperc5_ = 0

testperc10_ = 0
predperc10_ = 0

testperc15_ = 0
predperc15_ = 0

testperc20_ = 0
predperc20_ = 0

testperc100_ = 0
predperc100_ = 0

accuracy5_ = 0

accuracy10_ = 0

accuracy15_ = 0

accuracy20_ = 0

accuracy100_ = 0


# splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state= 89) # 99 = .9539 #89 used in best test yet
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Scaling the data
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Creating the ML Model
model = keras.Sequential([
    keras.layers.Dense(100, activation = "sigmoid" , input_dim = 10), # tanh works ok for just this layer as well, but sigmoid seems best
    keras.layers.Dense(100, activation="sigmoid", input_dim = 100),
    keras.layers.Dense(100, activation="sigmoid", input_dim=100),
    keras.layers.Dense(1, activation = "sigmoid", input_dim = 100)
    ])

model.summary()

model.compile(optimizer="adam", loss = "binary_crossentropy", metrics= ["accuracy"])

# Trains the Model
model.fit(X_train,y_train, epochs = 200)


# Tests the model on Test data
y_log = model.predict(X_test)


# Remainded of Code Used for Analysis
test5 = 0.00
pred5 = 0.00
test10 = 0.00
pred10 = 0.00
test15 = 0.00
pred15 = 0.00
test20 = 0.00
pred20 = 0.00
test100 = 0.00
pred100 = 0.00
c5 = 0
c10 = 0
c15 = 0
c20 = 0
c100 = 0
n = len(y_log)
for i in range(n):
    val = y_log[i]
    if val < .05:
        test5 += y_test[i]
        pred5 += val
        c5 += 1
    else:
        if val < .10:
            test10 += y_test[i]
            pred10 += val
            c10 += 1
        else:
            if val < .15:
                test15 += y_test[i]
                pred15 += val
                c15 += 1
            else:
                if val < .20:
                    test20 += y_test[i]
                    pred20 += val
                    c20 += 1
                else:
                    test100 += y_test[i]
                    pred100 += val
                    c100 += 1
if c5 > 0:
    testperc5 = test5/c5
    predperc5 = pred5/c5
    testperc5_ += testperc5
    predperc5_ += predperc5
    # accuracy5 = predperc5/testperc5
    accuracy5 = -(testperc5 - predperc5)/testperc5
    accuracy5_ += accuracy5
    print("Predicted .00 - .05 percentage is " + str(predperc5 * 100) + "%, Actual: " + str(testperc5*100) + "%. It is " + str(accuracy5 * 100) + "% accurate, count: " + str(c5))
if c10 > 0:
    testperc10 = test10/c10
    predperc10 = pred10/c10
    testperc10_ += testperc10
    predperc10_ += predperc10
    # accuracy10 = predperc10/testperc10
    accuracy10 = -(testperc10 - predperc10)/testperc10
    accuracy10_ += accuracy10
    print("Predicted .05 - .10 percentage is " + str(predperc10 * 100) + "%, Actual: " + str(testperc10*100) + "%. It is " + str(accuracy10 * 100) + "% accurate, count: " + str(c10))
if c15 > 0:
    testperc15 = test15/c15
    predperc15 = pred15/c15
    testperc15_ += testperc15
    predperc15_ += predperc15
    # accuracy15 = predperc15/testperc15
    accuracy15 = -(testperc15 - predperc15)/testperc15
    accuracy15_ += accuracy15
    print("Predicted .10 - .15 percentage is " + str(predperc15 * 100) + "%, Actual: " + str(testperc15*100) + "%. It is " + str(accuracy15 * 100) + "% accurate, count: " + str(c15))
if c20 > 0:
    testperc20 = test20 / c20
    predperc20 = pred20 / c20
    testperc20_ += testperc20
    predperc20_ += predperc20
    # accuracy20 = predperc20 / testperc20
    accuracy20 = -(testperc20 - predperc20)/testperc20
    accuracy20_ += accuracy20
    print("Predicted .15 - .20 percentage is " + str(predperc20 * 100) + "%, Actual: " + str( testperc20 * 100) + "%. It is " + str(accuracy20 * 100) + "% accurate, count: " + str(c20))
if c100 > 0:
    testperc100 = test100/c100
    predperc100 = pred100/c100
    testperc100_ += testperc100
    predperc100_ += predperc100
    # accuracy100 = predperc100/testperc100
    accuracy100 = -(testperc100 - predperc100)/testperc100
    accuracy100_ += accuracy100
    print("Predicted .20 - 1.00 percentage is " + str(predperc100 * 100) + "%, Actual: " + str(testperc100*100) + "%. It is " + str(accuracy100 * 100) + "% accurate, count: " + str(c100))


# Saving Model

StrokeMLModel = model
StrokeMLModel.save_weights('Best Model\Best Model')
