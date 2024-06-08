#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
breast_data = pd.read_csv('C:/Users/Hitesh/Downloads/data.csv')

# Preprocess the data
breast_data.replace({'diagnosis': {'M': 0, 'B': 1}}, inplace=True)
breast_data.drop('Unnamed: 32', axis=1, inplace=True)

# Split the data
X = breast_data.drop('diagnosis', axis=1)
Y = breast_data['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=2, test_size=0.2)

# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate the model
train_predictions = model.predict(x_train)
test_predictions = model.predict(x_test)
print("Training Accuracy:", accuracy_score(y_train, train_predictions))
print("Test Accuracy:", accuracy_score(y_test, test_predictions))


# In[ ]:





# ## Model after Scaling 

# In[3]:


#!/usr/bin/env python
# coding: utf-8

#import all libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#import dataset from local drive
breast_data = pd.read_csv('C:/Users/Hitesh/Downloads/data.csv')

#Infromation about Features and data we have for our model
print(breast_data.shape)

#Describe function gives information about means, min, max value we have for a particular feature
print(breast_data.describe())

# Check for missing values
print(breast_data.isna().sum())

# Group by diagnosis and calculate mean values for each group
print(breast_data.groupby('diagnosis').mean())

# Replace 'M' with 0 and 'B' with 1 in the 'diagnosis' column
breast_data.replace({'diagnosis': {'M': 0, 'B': 1}}, inplace=True)

# Drop the 'Unnamed: 32' column
breast_data.drop('Unnamed: 32', axis=1, inplace=True)

# Define features and labels
X = breast_data.drop('diagnosis', axis=1)
Y = breast_data['diagnosis']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=2, test_size=0.2)

# Scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict and calculate accuracy for the training set
train_predictions = model.predict(x_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

# Predict and calculate accuracy for the testing set
test_predictions = model.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test Accuracy:", test_accuracy)


# In[ ]:




