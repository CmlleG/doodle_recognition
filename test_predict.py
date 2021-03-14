import joblib

model = joblib.load(open('modelMOB.joblib', 'rb'))
print(type(model))