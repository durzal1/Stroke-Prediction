# Stroke-Prediction
Predict if patient is likely to have a stroke

However, due to the nature of the data, no individual prediction was greater than a 30% chance of a stroke since a large majority of the patients from the test data did not experience a stroke. Thus, this proves that having a stroke in itself is a rare occurence. So consider 30% chance a high chance for the sake of this model. 

# How to use

run StrokeGui.py

Input personal details and press submit

![image](https://user-images.githubusercontent.com/67489054/219912858-49ba4c5f-ba1e-4833-90da-88137ab21c65.png)


Your percent likelihood of stroke will be outputed to the screen. 

![image](https://user-images.githubusercontent.com/67489054/219913279-e3939a7b-4bed-4d7d-8773-f81dcb080458.png)

# Results

After we trained the ML Model, we used a 3d scatterplot to graph it and attempt to find two variables that together may signify a higher chance of strokes in patients.


Note: The Left will have a picture of our Models predictions while the right will have a picture of the true data.


Here are the results:

1) Age vs BMI 

![image](https://user-images.githubusercontent.com/67489054/219918285-22e0611a-1f1c-44fa-917b-7fd01a387ab3.png)

As you can see the older a person got, the more likely it was that they had a stroke.
There also seems to be a much weaker relationship between a higher BMI leading to a higher chance of stroke.
As people aged in this dataset, their BMI increased on average as well.


2) Age vs Average Glucose Levels

![image](https://user-images.githubusercontent.com/67489054/219918337-46d89cbb-d537-46c3-b3a1-45d1bb5a65a2.png)

Just like the previous graph, as the older a person got, the more likely they were to have a stroke.
It is also notable that people with higher glucose levels tended to be more likely to have a stroke as well. 
However, the fact that the slope of the Age-to-Predicted Stroke Percentage Graph was relatively constant across all BMI's indicate that in reality, BMI's affects are confounded with age's. 


3) BMI vs Average Glucose Levels 

![image](https://user-images.githubusercontent.com/67489054/219918359-cccdc647-cb1d-4692-91d7-6cd6b893d5fe.png)

Both BMI's and Avg Glucose Level's relationships with stroke prediction and incidence are scattered.
Neither of these have a strong impact on stroke occurence without by themselves or with each other.
They could still have a strong impact when combined with other variables not explored in these graphs.

