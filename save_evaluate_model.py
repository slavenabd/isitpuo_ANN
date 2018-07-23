# Author: slavenabd, sabdovic@gmail.com
# Artificial Neural Network (isitPUO)
# Predicting posterior urethral obstruction in boys with lower urinary tract symptoms
## modify, evaluate and save own model to the disk

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset. Save dataset to .csv file. Check Data_3.csv for example.
dataset = pd.read_csv('Data_3.csv')
X = dataset.iloc[:, :7].values
y = dataset.iloc[:, 7].values

# Splitting the data set into the Training set and Test set. Modify test sample size, current settings is 0.2 = 20%. Changing the value will yield different training sample size and infuence accuracy.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling. RobustScaler is chosen over StandardScaler. If different dataframe is used, maybe StandardScaler is optimal.
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# initialising the ANN
classifier = Sequential()

# The First Hidden Layer. Add more layers, change number of units. If overfitting occurs increase dropout. In high variance increase dropout gradually, never above 0.5
classifier.add(Dense(activation = 'relu', input_dim = 7, units = 4, kernel_initializer = 'uniform'))
classifier.add(Dropout(p = 0.0))

# The Second hidden layer
classifier.add(Dense(activation = 'relu', units = 4, kernel_initializer = 'uniform'))

# The Output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set. In different datafrem find optimal batch size, for more observations choose bigger value. Change number of epochs nad batch size according to tuning results.
classifier.fit(X_train, y_train, batch_size = 5, epochs = 50)

# Predicting the Test set results. In this model prediction is set at 80%. Change to adjust perforamnce.
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


# Save model to JSON and HDF5
from keras.models import model_from_json
import os

# evaluate the model
scores = classifier.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")

# To find loading code, open isitpuo_bt_load.py or isitpuo_st_load.py

# Evaluating the ANN (K-fold cross validation, Bias-Variance Tradeoff). Evaluate mean accuracy and variance.
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', input_dim = 7, units = 4, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = 'relu', units = 4, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 5, epochs = 50)
accuracies = cross_val_score(estimator = classifier, X = X_train_scaled, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Tuning the ANN. Search for the best parameters for the ANN.
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', input_dim = 7, units = 4, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = 'relu', units = 4, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [3, 5], 
              'epochs': [50, 75],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train_scaled, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_