# Author: slavenabd, sabdovic@gmail.com
# Artificial Neural Network (isitPUO)
# Predicting posterior urethral obstruction in boys with lower urinary tract symptoms
# version _st : single sample terminal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Input variables
A = input("Age (years.0): ")
PF = input("Peak Flow rate (mL/s): ")
TP = input("Time to Peak (s): ")
V = input("Volume (mL): ")
FT = input("Flow Time (s): ")
VT = input("Voiding Time (s): ")
AF = input("Average Flow rate (mL/s): ")

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

# initialising the ANN
classifier = Sequential()

# The First Hidden Layer
classifier.add(Dense(activation = 'relu', input_dim = 7, units = 4, kernel_initializer = 'uniform'))

# The Second hidden layer
classifier.add(Dense(activation = 'relu', units = 4, kernel_initializer = 'uniform'))

# The Output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set (input, ground truth)
classifier.fit(X_train, y_train, batch_size = 5, epochs = 50)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.80)

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
      '-----------------------------------','\n')

# Predicting a single new observation
""""[Age,PeakFlowrate,TimeToPeak,Volume,FlowTime,VoidingTime,AverageFlowrate]"""""
new_prediction = classifier.predict(sc.transform(np.array([[A, PF, TP, V, FT, VT, AF]])))
new_prediction1 = (new_prediction > 0.80)
print('\n'
      'Input variables:','\n'
      '-----------------------------------','\n'
      'Age: ', A, 'years','\n'
      'Peak flow rate: ', PF, 'mL/s','\n'
      'Time to peak: ', TP, 's','\n'
      'Volume: ', V, 'mL','\n'
      'Flow time: ', FT, 's','\n'
      'Voiding time: ', VT, 's','\n'
      'Average flow rate: ', AF, 'mL/s','\n'
      '-----------------------------------','\n'
      "Probability of PUO:")
print(new_prediction[0][0])
print(new_prediction1[0][0],"\n"
      '-----------------------------------')
plt.pie(new_prediction, colors='rd', autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title("Probability of PUO")
plt.tight_layout()
plt.show()

print("\n")
input('Press enter to exit...')