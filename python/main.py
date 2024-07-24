# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:00:43 2024

@author: Alp Altunsoy
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def read_data():
    print("Importing Csv train and test files")
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    return train_data, test_data

def scaling(x):
    
    print("X scaling is started")
    sc = StandardScaler()
    X = sc.fit_transform(x)
    
    return X

def make_encoding_x(train_data):
    print("Encoding independent variables")
    #encoding some columns
    le = LabelEncoder()
    customer_types = le.fit_transform(train_data.iloc[:,3])
    type_of_travel = le.fit_transform(train_data.iloc[:,5])
    gender = le.fit_transform(train_data.iloc[:,2])

    #encoding some columns with one hat encoding
    ohe = OneHotEncoder()
    class_=ohe.fit_transform(train_data.iloc[:,6:7]).toarray()
    return customer_types,type_of_travel,gender,class_
    
def make_x(train_data, customer_types,type_of_travel,gender,class_):
    print("Creating X variables")    
    #creating x data_sets for svm
    gender = pd.DataFrame(data = gender, columns=["Gender"])
    customer_types = pd.DataFrame(data = customer_types, columns=["Customer Type"])
    x = pd.concat([gender, customer_types,train_data.iloc[:,4:5]],axis = 1)
    type_of_travel = pd.DataFrame(data = type_of_travel, columns=["Type of Travel"])
    class_ = pd.DataFrame(data= class_ , columns = ["Business", "Eco","Eco Plus"])
    x = pd.concat([x,type_of_travel, class_,train_data.iloc[:,7:24]], axis =1)
    return x

def make_y(train_data):
    print("Making dependent Variables")
    y = train_data.iloc[:,24]
    return y 


def missing_imputer(x):
    print("Imputing missing values")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_imputed = imputer.fit_transform(x)
    return x_imputed

def svm_classifier(x_train,y_train,x_test):
    svc = SVC(kernel = "rbf")
    print("SVC learning is started")
    svc.fit(x_train,y_train)
    print("SVC prediction is started")
    y_pred =svc.predict(x_test) 
    print("Prediction is finished")
    return y_pred


def start():
    train_data, test_data = read_data()
    customer_types,type_of_travel,gender,class_ = make_encoding_x(train_data)
    
    #train files making
    x_train = make_x(train_data, customer_types,type_of_travel,gender,class_)
    y_train = make_y(train_data)
    #test files making
    customer_types,type_of_travel,gender,class_ = make_encoding_x(test_data)
    x_test = make_x(test_data, customer_types,type_of_travel,gender,class_)
    y_test = make_y(test_data)
    
    
    
    #missing values
    x_train = missing_imputer(x_train)
    x_test = missing_imputer(x_test)
    
    
    #scaling datas 
    x_test = scaling(x_test)
    x_train = scaling(x_train)
    
    #test files making
    y_pred = svm_classifier(x_train, y_train, x_test)
    
    
    return x_train, y_train,x_test,y_test,y_pred


def main():
    global x_train, y_train,x_test,y_test,y_pred
    x_train, y_train,x_test,y_test,y_pred = start()
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    
    

if __name__ == "__main__":
    main()
