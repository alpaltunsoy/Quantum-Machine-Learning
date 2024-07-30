# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:03:11 2024

@author: Alp Altunsoy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:55:02 2024

@author: Alp Altunsoy
"""
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import time
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json
import boto3
from datetime import datetime


def load_data():
    print("Importing CSV train and test files")
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    return train_data, test_data

def minimize_rows(data, index,name):
    print("Minimizing rows of ",str(name))
    data = data.head(index)
    return data

def encoding_and_creating_sets_for_x(data):
   print("Encoding independent variables")
   #encoding some columns
   le = LabelEncoder()
   gender = le.fit_transform(train_data.iloc[:,2])
   gender = pd.DataFrame(data = gender, columns=["Gender"])
   x = pd.concat([gender, data.iloc[:,7:8],data.iloc[:,22:24],data.iloc[:,14:15]], axis =1)
   return x
    
def make_y(y):
    print("Making dependent Variables")
    y = y.iloc[:,-1].values
    return y 
    
def upload_to_s3(file_name, bucket_name, object_name=None):

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3 = boto3.resource('s3')
    try:
        s3.Bucket(bucket_name).upload_file(file_name, object_name)
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False
    return True


def missing_values(x):
    print("Imputing missing values")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_imputed = imputer.fit_transform(x)
    return x_imputed
    
def scaling_data(x):
    print("X scaling is started")
    sc = MinMaxScaler()
    x = sc.fit_transform(x)
    return x
def y_encoding(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y 
    

def quantum_algorithm(x_train,y_train,x_test,y_test):
    
    print("Quantum algorithm is started")
    print("Finding features")
    
    num_features = x_train.shape[1]
    
    #drawing circuit diagram
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    feature_map.decompose().draw(output="mpl", style="clifford", fold=20)
    
    
    print("Drawing ansatz circuit")
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
        
    vqc = VQC(sampler=sampler,feature_map=feature_map,ansatz=ansatz,optimizer=optimizer,callback=callback_graph,)

    # clear objective value history
    objective_func_vals = []

    start = time.time()
    print("Learning is started")
    vqc.fit(x_train, y_train)
    elapsed = time.time() - start

    print(f"Training time: {round(elapsed)} seconds")

    train_score_q4 = vqc.score(x_train, y_train)
    test_score_q4 = vqc.score(x_test, y_test)

    print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
    print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")
    
    # Save scores to a JSON file
    scores = {
       "train_score": train_score_q4,
       "test_score": test_score_q4,
       "training time" : elapsed
   }
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = current_time + "_report.json"
    with open(report_name, "w") as f:
        json.dump(scores, f)

    print("Scores have been saved to 'vqc_scores.json'")

    
    bucket_name = 'alposmanyeni'
     # Include date and time in the file name

    if upload_to_s3(report_name, bucket_name,report_name):
        print(f"Report successfully uploaded to S3 at '{bucket_name}/{report_name}'")
    else:
        print("Failed to upload the report to S3.")




def flow():
    
    #rows number of train and test values
    test_number = 100
    train_number = 2000
    
    global x_train, x_test, y_train,y_test
    global train_data, test_data
    
    train_data, test_data = load_data()
    x_train = encoding_and_creating_sets_for_x(train_data)
    x_test =  encoding_and_creating_sets_for_x(test_data)

    #creating x test and x train
    x_train = minimize_rows(x_train, train_number,"train dataset")
    x_test = minimize_rows(x_test, test_number, "test dataset")
    #creating y test and y train
    y_train = make_y(train_data)
    y_test = make_y(test_data)
    
    #minimizing rows of y sets
    y_test = y_test[0:test_number]
    y_train = y_train[0:train_number]

    #encoding y datas
    y_train = y_encoding(y_train)
    y_test = y_encoding(y_test)
    
    #scaling data
    x_train = scaling_data(x_train)
    x_test = scaling_data(x_test)
    
    #adding missing values
    x_train = missing_values(x_train)
    x_test = missing_values(x_test)

    #starting quantum machine learning
    quantum_algorithm(x_train,y_train,x_test,y_test)
   
def main():
    flow()

if __name__ == "__main__":
    main()