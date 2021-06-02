import pickle as pkl
import sklearn
import numpy as np
from sklearn.utils.extmath import weighted_mode 
# 57	0	0	140	241	0	1	123	1	0.2	1	0	3	0
# age = 57
# sex = 0
# cp = 0
# trestbps = 140
# chol = 241
# fbs = 0
# restecg = 1
# thalach = 123
# exang = 1
# oldpeak = 0.2
# slope = 1
# ca = 0
# thal = 3

# 57	0	0	120	354	0	1	163	1	0.6	2	0	2	1
age = 57
sex = 0
cp = 0
trestbps = 120
chol = 354
fbs = 0
restecg = 1
thalach = 163
exang = 1
oldpeak = 0.6
slope = 2
ca = 0
thal = 2

def query():
    with open('./models/scaler.pkl', 'rb') as f:
        loaded_scaler = pkl.load(f)
    with open('./models/heart_disease_classifier.pkl', 'rb') as f:
        loaded_clf = pkl.load(f)
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    print(data)
    scaled_data = loaded_scaler.transform(data)
    print(scaled_data)
    results = loaded_clf.predict(scaled_data)
    return str(results)



output = query()
print(output)