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
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from datetime import datetime


def load_data():
    print("Importing CSV train and test files")
    train_data = pd.read_csv(r"D:\DOSYALAR\Kodlar\Quantum-Machine-Learning\python\train.csv")
    test_data = pd.read_csv(r"D:\DOSYALAR\Kodlar\Quantum-Machine-Learning\python\test.csv")
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
    



def start(svm_type):
    
    #rows number of train and test values
    test_number = 200
    train_number = 2500
    
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
    #test files making
    y_pred = svm_classifier(x_train, y_train, x_test,svm_type)

    #starting quantum machine learning
    return x_train, y_train,x_test,y_test,y_pred,svm_type

def svm_classifier(x_train,y_train,x_test,svm_type):
    svc = SVC(kernel = str(svm_type))
    print("SVC learning is started")
    svc.fit(x_train,y_train)
    print("SVC prediction is started")
    y_pred =svc.predict(x_test) 
    print("Prediction is finished")
    return y_pred

def calculator(cm):
    print("Calculating Metrics\n")
    TP, FP, FN, TN = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    specificity = TN / (TN + FP)
    return accuracy,precision,recall,f1_score,specificity

def flow():
    all_results = []
    svm_type = ['linear',"rbf","sigmoid"]
    
    for svm_types in svm_type: 
        #taking time for calculating time
        start_time = time.time()
        print("Making predictions for ", svm_types)
    
        #defining variables for independent and dependent variables and starting program
        x_train, y_train,x_test,y_test,y_pred,svm_type = start(svm_types)
    
        #making confusion matrix for analysis
        cm = confusion_matrix(y_test,y_pred)
        print(cm)
        accuracy,precision,recall,f1_score,specificity = calculator(cm)
        
        #finishing time for calculating elapsed time
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        
        result = {
           "kernel_type": svm_type,
           "execution_time_ms": elapsed_time,
           'metrics': {
               'accuracy': accuracy,
               'precision': precision,
               'recall': recall,
               'f1_score': f1_score,
               'specificity': specificity
           }
       }
        all_results.append(result)
        
    
    
    report(all_results)
    print("SVC type is completed \n\n\n")

def report(result):
    with open(r'D:\DOSYALAR\Kodlar\Quantum-Machine-Learning\python\report_new.json', 'w') as f:
        json.dump(result, f, indent=4)
    print("Report saved to report_new.json")


def main():
    flow()

if __name__ == "__main__":
    main()