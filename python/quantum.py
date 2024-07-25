# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:00:43 2024

@author: Alp Altunsoy
"""

import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import time
import json
from qiskit.circuit.library import ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from IPython.display import clear_output
import time
from qiskit_machine_learning.algorithms.classifiers import VQC
import matplotlib.pyplot as plt



# Importing CSV train and test files
print("Importing CSV train and test files")
train_data = pd.read_csv(r"D:\DOSYALAR\Kodlar\quantum-computing-speed-comparison\python\train.csv")
test_data = pd.read_csv(r"D:\DOSYALAR\Kodlar\quantum-computing-speed-comparison\python\test.csv")

# Drop the 12th column (index 11) from train_data and test_data
drop_index = 11
train_data = train_data.drop(columns=[train_data.columns[drop_index]])
test_data = test_data.drop(columns=[test_data.columns[drop_index]])

# Drop the 12th column (index 11) from train_data and test_data
drop_index = 19
train_data = train_data.drop(columns=[train_data.columns[drop_index]])
test_data = test_data.drop(columns=[test_data.columns[drop_index]])


train_data = train_data.head(50)
test_data = test_data.head(20)




# Check the shape of the data after dropping the column
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Encoding independent variables for training data
print("Encoding independent variables")
le = LabelEncoder()
customer_types = le.fit_transform(train_data.iloc[:, 3])
type_of_travel = le.fit_transform(train_data.iloc[:, 5])
gender = le.fit_transform(train_data.iloc[:, 2])

ohe = OneHotEncoder()
class_ = ohe.fit_transform(train_data.iloc[:, 6:7]).toarray()

# Creating X variables for training data
print("Creating X variables")
gender_df = pd.DataFrame(data=gender, columns=["Gender"])
customer_types_df = pd.DataFrame(data=customer_types, columns=["Customer Type"])
type_of_travel_df = pd.DataFrame(data=type_of_travel, columns=["Type of Travel"])
class_df = pd.DataFrame(data=class_, columns=["Business", "Eco", "Eco Plus"])

# Adjust column selection after dropping a column
x_train = pd.concat([gender_df, customer_types_df, train_data.iloc[:, 4:5], type_of_travel_df, class_df, train_data.iloc[:, 7:22]], axis=1)

# Making dependent variables for training data
print("Making dependent Variables")
y_train = train_data.iloc[:, 22].values

# Encoding independent variables for test data
customer_types = le.fit_transform(test_data.iloc[:, 3])
type_of_travel = le.fit_transform(test_data.iloc[:, 5])
gender = le.fit_transform(test_data.iloc[:, 2])
class_ = ohe.transform(test_data.iloc[:, 6:7]).toarray()

# Creating X variables for test data
gender_df = pd.DataFrame(data=gender, columns=["Gender"])
customer_types_df = pd.DataFrame(data=customer_types, columns=["Customer Type"])
type_of_travel_df = pd.DataFrame(data=type_of_travel, columns=["Type of Travel"])
class_df = pd.DataFrame(data=class_, columns=["Business", "Eco", "Eco Plus"])

# Adjust column selection after dropping a column
x_test = pd.concat([gender_df, customer_types_df, test_data.iloc[:, 4:5], type_of_travel_df, class_df, test_data.iloc[:, 7:22]], axis=1)

# Making dependent variables for test data
y_test = test_data.iloc[:, 22].values

# Imputing missing values for both train and test data
print("Imputing missing values")
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# Scaling data for both train and test data
print("X scaling is started")
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Number of features in the dataset after dropping the column
num_features = x_train.shape[1]
# Create the ZZFeatureMap with the updated number of features
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
# Decompose and draw the quantum circuit
feature_map.decompose().draw(output="mpl", style="clifford", fold=20)

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", style="clifford", fold=20)

optimizer = COBYLA(maxiter=100)

sampler = Sampler()


objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()
    
    
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []

print("Buraya geldi")
start = time.time()
vqc.fit(x_train, y_train)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

train_score_q4 = vqc.score(x_train, y_train)
test_score_q4 = vqc.score(x_test, y_test)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")



print("Main function is finished")
