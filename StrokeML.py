import pandas as pd
import numpy as np
# import tensorflow

data = pd.read_csv("brain_stroke.csv")

print(data.columns)

data.drop(['work_type'], axis=1)

print(data.columns);
