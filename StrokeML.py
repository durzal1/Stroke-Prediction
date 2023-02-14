import pandas as pd

print("hello")
data = pd.read_csv("brain_stroke.csv")

print('TEst')
print(data.columns)
#data.drop(['work_type'], axis=1) cmd isn't working for some reason
#print(data.columns)