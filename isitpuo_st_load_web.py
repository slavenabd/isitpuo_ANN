# Author: slavenabd, sabdovic@gmail.com
# Artificial Neural Network (isitPUO)
# Predicting posterior urethral obstruction in boys with lower urinary tract symptoms
# version _st_load_web : single sample terminal from saved NN model for web

# Input variables
A = input("Age (years.0): ")
PF = input("Peak Flow rate (mL/s): ")
TP = input("Time to Peak (s): ")
V = input("Volume (mL): ")
FT = input("Flow Time (s): ")
VT = input("Voiding Time (s): ")
AF = input("Average Flow rate (mL/s): ")

import numpy as np
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

with open("Output.txt", "w") as text_file:
    print('Age: {}'.format(A), 'years','\n'
          'Peak flow rate: {}'.format(PF), 'mL/s','\n'
          'Time to peak: {}'.format(TP), 's','\n'
          'Volume: {}'.format(V), 'mL','\n'
          'Flow time: {}'.format(FT), 's','\n'
          'Voiding time: {}'.format(VT), 's','\n'
          'Average flow rate: {}'.format(AF), 'mL/s','\n'
          "Probability of PUO: {}".format(new_prediction[0][0]), '\n'
          'Posterior urethral obstruction: {}'.format(new_prediction1[0][0]), file=text_file)
    
import gviz_api
description = {'name': ('string', 'Prediction'), 
               'result': ('number', 'Result')}
data = {'name': 'Prediction', 'result': new_prediction}
data_table = gviz_api.DataTable(description)
data_table.LoadData(data)
return data_table.ToJSon()