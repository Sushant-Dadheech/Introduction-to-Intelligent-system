"""
Bayesian Network Practical
Using pgmpy to model relationships between Study, Sleep, Stress, and Exam Pass.

Note: Requires pgmpy (pip install pgmpy)
"""

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Sample dataset
data = pd.DataFrame({
    'Study':    ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No'],
    'Sleep':    ['Yes', 'No', 'Yes', 'No', 'No',  'Yes', 'No', 'Yes', 'Yes','No'],
    'Stress':   ['No',  'Yes','No',  'Yes','Yes',  'No',  'Yes','No',  'No', 'Yes'],
    'ExamPass': ['Yes', 'No', 'Yes', 'No', 'No',  'Yes', 'No', 'Yes', 'Yes','No']
})

print("Sample Data:")
print(data)

# Define the Bayesian Network structure
model = BayesianNetwork([
    ('Study', 'Stress'),    # Studying reduces stress
    ('Sleep',  'Stress'),   # Sleep reduces stress
    ('Stress', 'ExamPass')  # Stress affects exam outcome
])

# Train using Maximum Likelihood Estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)
print("\nModel successfully trained!")
print("Edges in the model:", model.edges())

# --- Inference Queries ---
inference = VariableElimination(model)

q1 = inference.query(variables=['ExamPass'], evidence={'Stress': 'Yes'})
print("\nP(ExamPass | Stress=Yes):")
print(q1)

q2 = inference.query(variables=['Stress'], evidence={'Study': 'No', 'Sleep': 'No'})
print("\nP(Stress | Study=No, Sleep=No):")
print(q2)

q3 = inference.query(variables=['ExamPass'], evidence={'Study': 'Yes', 'Sleep': 'Yes'})
print("\nP(ExamPass | Study=Yes, Sleep=Yes):")
print(q3)
