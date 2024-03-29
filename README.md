# Heart-Disease (ml)

This is a simple AI Machine Learning Model that can predict the liklihood of heart disease in a patient depending on a number of factors. Data used can be found in Data Directory.
Please bear in mind this is just a very simple AI (85% accuracy) and should not substitute doctor's advice.
You can refer to query.py for a sample script that can be run in python, incorporating the pre-trained model found in the models folder.
Make sure to run `pip install -r requirements.txt` in a terminal beforehand!
Enjoy!

## Factors
age - age in years
sex - sex (1 = male; 0 = female)
cp - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
trestbps - resting blood pressure (in mm Hg on admission to the hospital)
chol - serum cholestoral in mg/dl
fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
restecg - resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)
thalach - maximum heart rate achieved
exang - exercise induced angina (1 = yes; 0 = no)
oldpeak - ST depression induced by exercise relative to rest
slope - the slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)
ca - number of major vessels (0-3) colored by flourosopy
thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
target - the predicted attribute - diagnosis of heart disease (angiographic disease status) (Value 0 = < 50% diameter narrowing; Value 1 = > 50% diameter narrowing)

## Acknwledgements
Creators:

Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

https://archive.ics.uci.edu/ml/datasets/Heart+Disease

Donor: 
David W. Aha (aha '@' ics.uci.edu) (714) 856-8779
