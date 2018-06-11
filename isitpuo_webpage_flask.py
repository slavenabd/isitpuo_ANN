# Author: slavenabd, sabdovic@gmail.com
# Artificial Neural Network (isitPUO)
# Predicting posterior urethral obstruction in boys with lower urinary tract symptoms
# version _webpage_flask : single sample webpage evaluation from saved NN model

import numpy as np

def prepare_classification():
  global sc
  global loaded_model

  import pandas as pd
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
  loaded_model.load_weights('model.h5')

  # evaluate loaded model on test data
  loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



from flask import Flask
app = Flask(__name__)
app.before_first_request(prepare_classification)

from flask import request
from flask import render_template


FEATURE_NAMES = ['age', 'peak_flow_rate', 'time_to_peak', 'volume', 'flow_time', 'voiding_time', 'average_flow_rate']

@app.route('/')
def default():
    return render_template('isitpuo_webpage_flask_template.html', **{feature_name: '' for feature_name in FEATURE_NAMES})

@app.route('/evaluate', methods=['GET'])
def evaluate():
    param_values = {feature_name: request.args.get(feature_name, type=float) for feature_name in FEATURE_NAMES}
    features = np.array([[param_values[feature_name] for feature_name in FEATURE_NAMES]])
    param_values['probability'] = loaded_model.predict(sc.transform(features))[0][0]
    param_values['prediction'] = param_values['probability'] > 0.8
    return render_template('isitpuo_webpage_flask_template.html', **param_values)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
