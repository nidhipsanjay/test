import joblib
import json

print("Extracting harmless statistical intercepts...")
# Load the model one last time
model = joblib.load('master_33_cancer_model.pkl')

# Extract the 33 numbers and cast them to native Python floats
intercepts = {i: float(model.estimators_[i].intercept_[0]) for i in range(33)}

# Save them to a lightweight dictionary
with open('clinical_intercepts.json', 'w') as f:
    json.dump(intercepts, f)

print("Saved to clinical_intercepts.json! You can now safely delete the .pkl file.")