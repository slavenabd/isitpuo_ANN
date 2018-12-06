# isitpuo_ANN
**Predicting late-presenting posterior urethral valve**

## About
*isitpuo ANN* is a deep artificial neural network that classifies uroflowmetry reports and predicts posterior urethral obstruction in boys with lower urinary tract symptoms (LUTS). The ground truth observations are cases of the posterior urethral valve in boys aged 3 to 17 years who presented with storage or voiding LUTS. Patients who persistently had peak flow rate below the 5th percentile for age and gender, and who were completely or partially unresponsive to the measures of standard urotherapy were referred for cystoscopy. Uroflow reports from patients prior to visualization and electroresection of PUV were retrospectively collected as ground truth observations for posterior urethral obstruction. Observations classified as normal were uroflow reports of children without LUTS, without anomalies of kidneys or urinary tract, without bladder outlet obstruction, neurogenic bladder or constipation.

## Disclaimer
This deep artificial neural network is for a research use only. Differential medical diagnosis and diagnostic workup should be guided by signs and symptoms, clinical course, medical history, and appropriate diagnostic tests according to the accepted guidelines. This neural network model is not validated as a diagnostic test for bladder outlet obstructions in both children and adults and should not be used as such.

## Files
+ *_st* is for the classification of single observations
+ *_bt* is for the classification of the batch of observations written in the *Batch.csv*
+ *Data_3.csv* are ground truth observations used for the ANN training

## Web application
Web application is available [here](https://isitpuo.herokuapp.com "isitpuo web app").

## Published paper
This work is published in World Journal of Urology; the full article is available [here](https://link.springer.com/epdf/10.1007/s00345-018-2588-9?author_access_token=zcdOEwBhm6ljGPRHd8l25Pe4RwlQNchNByi7wbcMAY4saoFb4UdT4u5DHP8E48UWqhEJkZ7ViunGSBZTA9awWbixwHKnV4jNw95v4Wkf_4nnn1KwbVboTrMgXvp0gv15qoFLVFyDRHDet8RsnIGjUg%3D%3D).
