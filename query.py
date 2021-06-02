import pickle as pkl
import sklearn
import numpy as np 

age = 55
sex = 1
cp = 0
trestbps = 140
chol = 217
fbs = 0
restecg = 1
thalach = 111
exang = 1
oldpeak = 5.6
slope = 0
ca = 0
thal = 3

def query():
    with open('./models/heart_disease_classifier.pkl', 'rb') as f:
        loaded_clf = pkl.load(f)
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    print(data)
    results = loaded_clf.predict(data)
    return str(results)


output = query()
print(output)