# Author: slavenabd, sabdovic@gmail.com
# Artificial Neural Network (isitpuo)
# Predicting posterior urethral obstruction in boys with lower urinary tract symptoms
# dataframe: uroflow report
# version _bt : batch samples terminal from saved NN model

import numpy as np
import pandas as pd
import matplotlib as plt
import keras
from keras.models import model_from_json

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

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print('\n')
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# Making predictions and evaluating the model
# Predicting the Test set results
y_pred = loaded_model.predict(X_test)
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
analyse_y_pred = loaded_model.predict(analyse_X_scaled)
Id = analyse_dataset.iloc[:,:1].values
output = np.concatenate([Id, analyse_y_pred], axis = 1)

from tabulate import tabulate
headers = ["Id", "PUO prediction"]
table = tabulate(output, headers, tablefmt="fancy_grid")
print(table)
print("\n")

input('Press enter to exit...')