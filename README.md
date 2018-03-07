# isitpuo_ANN
**Predicting posterior urethral obstruction in boys with lower urinary tract symptoms using deep artificial neural network**

## About
*isitpuo ANN* is a deep artifical neural network that classifies uroflowmetry reports and predicts posterior urethral obstruction (PUO) in boys wtih lower urinary tract symptoms (LUTS). The ground truth observations are cases of PUO in boys aged 3 to 18 years who presented with storage or voding LUTS. Patients who persitently had peak flow rate below 5th percentile for age and gender, and who were completely or partialy unresponsive to the measures of standard urotherapy were examined by urologist and referred for cistoscopy. Uroflow reports from patients prior to visualisation and electroresection of PUO were retropectively collected as ground truth observations for PUO. Observtions classified as normal were uroflow reports of children without LUTS, without anomalies of kidneys or urinary tract, withut bladder outlet obstruction, neurognic bladder or constipation.

## Disclaimer
This deep artificial neural network is for a reasearch use only. Differential medical diagnosis and diagnostic workup should be guided by signs and symptoms, clinical course, medical history, and appropriate diagnostic tests according to the accepted guidelines. This neural network model is not validated as a diagnostic test for bladder outlet obstructions in both children and adults and should not be used as such.

## File descriptions
_isitpuo_st_load.py_ = prediction from single new observation using the best saved model
_isitpuo_bt_load.py_ = prediction from batch of observations using the best saved model
_isitpuo_st.py_ = prediction from single new observation, each new prediction trains NN
_isitpuo_bt.py_ = prediction from batch of observationa, each new bath file load trains NN
_Data_3.csv_ = dataframe, ground truth observations
_Batch.csv_ = sample batch file of observations
