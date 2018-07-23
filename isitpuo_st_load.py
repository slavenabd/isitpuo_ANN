# Author: slavenabd, sabdovic@gmail.com
# Artificial Neural Network (isitPUO)
# Predicting posterior urethral obstruction in boys with lower urinary tract symptoms
# version _st_load : single sample terminal from saved NN model

# Input variables
A = input("Age (years.0): ")
PF = input("Peak Flow rate (mL/s): ")
TP = input("Time to Peak (s): ")
V = input("Volume (mL): ")
FT = input("Flow Time (s): ")
VT = input("Voiding Time (s): ")
AF = input("Average Flow rate (mL/s): ")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

# Predicting a single new observation
""""[Age,PeakFlowrate,TimeToPeak,Volume,FlowTime,VoidingTime,AverageFlowrate]"""""
new_prediction = loaded_model.predict(sc.transform(np.array([[A, PF, TP, V, FT, VT, AF]])))
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