# Author: slavenabd, sabdovic@gmail.com
# Artificial Neural Network (isitpuo)
# Predicting posterior urethral obstruction in boys with lower urinary tract symptoms
# dataframe: uroflow report
# version _bt : batch samples terminal

import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data_3.csv')
X = dataset.iloc[:, :7].values
y = dataset.iloc[:, 7].values

# Splitting the data set into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - standardization
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# The First Hidden Layer
classifier.add(Dense(activation = 'relu', input_dim = 7, units = 4, kernel_initializer = 'uniform'))

# The Second Hidden Layer
classifier.add(Dense(activation = 'relu', units = 4, kernel_initializer = 'uniform'))

# The Output Layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 5, epochs = 50)

# Making predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.8)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Performance report
Accuracy = (cm[0,0]+cm[-1,-1])/(cm[0,0]+cm[0,-1]+cm[-1,0]+cm[-1,-1])
Sensitivity = cm[-1,-1]/(cm[-1,-1]+cm[0,-1]) #Given the disease is present, the proability the test will be positive
Specificity = cm[0,0]/(cm[0,0]+cm[-1,0]) #Given the disease is absent, the probbility the test will be negative
PPV = cm[-1,-1]/(cm[-1,-1]+cm[-1,0]) #Given the test is positive, the probability the disease will be present
NPV = cm[0,0]/(cm[0,0]+cm[0,-1]) #Given the test is negative, the probability the disease is absent#

print('\n',
      '\n'
      '-----------------------------------','\n'
      'REPORT:','\n'
      '-----------------------------------','\n'
      'Prediction set at >=0.80','\n'
      '-----------------------------------','\n'
      'Accuracy: ', Accuracy,'\n'
      'Sensitivity: ', Sensitivity,'\n'
      'Specificity: ', Specificity,'\n'
      'Positive predictive value: ', PPV,'\n'
      'Negative predictive value: ', NPV,'\n'
      '-----------------------------------','\n',
      '\n')


# Predicting a batch of observations
analyse_dataset = pd.read_csv('Batch.csv')
analyse_X = analyse_dataset.iloc[:, 1:].values
analyse_X_scaled = sc.fit_transform(analyse_X)
analyse_y_pred = classifier.predict(analyse_X_scaled)
Id = analyse_dataset.iloc[:,:1].values
output = np.concatenate([Id, analyse_y_pred], axis = 1)

from tabulate import tabulate
headers = ["Id", "PUO prediction"]
table = tabulate(output, headers, tablefmt="fancy_grid")
print(table)
print("\n")

input('Press enter to exit...')