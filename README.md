# isitpuo_ANN
**Predicting late-presenting posterior urethral valve**

## About
*isitpuo ANN* is a deep artifical neural network that classifies uroflowmetry reports and predicts posterior urethral obstruction in boys wtih lower urinary tract symptoms (LUTS). The ground truth observations are cases of posterior urethral valve in boys aged 3 to 17 years who presented with storage or voding LUTS. Patients who persitently had peak flow rate below 5th percentile for age and gender, and who were completely or partialy unresponsive to the measures of standard urotherapy were referred for cistoscopy. Uroflow reports from patients prior to visualisation and electroresection of PUV were retropectively collected as ground truth observations for posterior urethral obstruction. Observtions classified as normal were uroflow reports of children without LUTS, without anomalies of kidneys or urinary tract, withut bladder outlet obstruction, neurognic bladder or constipation.

## Disclaimer
This deep artificial neural network is for a reasearch use only. Differential medical diagnosis and diagnostic workup should be guided by signs and symptoms, clinical course, medical history, and appropriate diagnostic tests according to the accepted guidelines. This neural network model is not validated as a diagnostic test for bladder outlet obstructions in both children and adults and should not be used as such.

## Files
+ *_st* is for the classification of single observations
+ *_bt* is for the classification of the batch of observations written in the *Batch.csv*
+ *Data_3.csv* are ground truth observations used for the ANN training

## Web application
Web application is available [here](https://isitpuo.herokuapp.com "isitpuo web app").
