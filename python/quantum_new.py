# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:55:02 2024

@author: Alp Altunsoy
"""
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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


def load_data():
    print("Importing CSV train and test files")
    train_data = pd.read_csv(r"D:\DOSYALAR\Kodlar\quantum-computing-speed-comparison\python\train.csv")
    test_data = pd.read_csv(r"D:\DOSYALAR\Kodlar\quantum-computing-speed-comparison\python\test.csv")
    return train_data, test_data

def minimize_rows(data, index,name):
    print("Minimizing rows of ",str(name))
    data = data.head(index)
    return data

def encoding_and_creating_sets_for_x(data):
   print("Encoding independent variables")
   #encoding some columns
   le = LabelEncoder()
   customer_types = le.fit_transform(train_data.iloc[:,3])
   type_of_travel = le.fit_transform(train_data.iloc[:,5])
   gender = le.fit_transform(train_data.iloc[:,2])

   #encoding some columns with one hat encoding
   ohe = OneHotEncoder()
   class_=ohe.fit_transform(train_data.iloc[:,6:7]).toarray()
   print("Creating X variables")    
   #creating x data_sets for svm
   gender = pd.DataFrame(data = gender, columns=["Gender"])
   customer_types = pd.DataFrame(data = customer_types, columns=["Customer Type"])
   class_ = pd.DataFrame(data= class_ , columns = ["Business", "Eco","Eco Plus"])
   type_of_travel = pd.DataFrame(data = type_of_travel, columns=["Type of Travel"])
   x = pd.concat([gender, customer_types, data.iloc[:,4:5]],axis = 1)
  
   x = pd.concat([x,type_of_travel, class_,train_data.iloc[:,7:-1]], axis =1)
   return x
    
def make_y(y):
    print("Making dependent Variables")
    y = train_data.iloc[:,-1].values
    return y 
    
    
def feature_minimize(train_data, test_data, drop_index):

    train_data = train_data.drop(columns=[train_data.columns[drop_index]])
    test_data = test_data.drop(columns=[test_data.columns[drop_index]])
    print(str(drop_index), ".th index columns are removed")
    return train_data, test_data

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
    global num_features
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


def flow():
    global train_data, test_data 
    train_data, test_data = load_data()
    train_data, test_data = feature_minimize(train_data, test_data , 11)
    train_data, test_data = feature_minimize(train_data, test_data, 12)
    train_data = minimize_rows(train_data, 50,"train dataset")
    test_data = minimize_rows(test_data, 10, "test dataset")
    
    print("Now current train_dataset is ", str(train_data.shape))
    print("Now current test_dataset is ", str(test_data.shape))
    global x_train, x_test
    x_train = encoding_and_creating_sets_for_x(train_data)
    x_test = encoding_and_creating_sets_for_x(test_data)
    global y_test, y_train
    y_train = make_y(train_data)
    y_test = make_y(test_data)
    y_train = y_encoding(y_train)
    y_test = y_encoding(y_test)
    
    x_train = missing_values(x_train)
    x_test = missing_values(x_test)
    x_train = scaling_data(x_train)
    x_test = scaling_data(x_test)
    
    quantum_algorithm(x_train,y_train,x_test,y_test)
   
    


def main():
    flow()

if __name__ == "__main__":
    main()