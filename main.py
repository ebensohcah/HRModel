
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import pickle

import warnings
warnings.filterwarnings("ignore")

# Read the train file
train_file = pd.read_csv('Human_Resources.csv')

# Display first 5 records from the file
print(train_file.head())

# drop the null values
train_file.dropna(axis=0, inplace=True)

# Understand the structure of the datasets
print('The size of the train file:', train_file.shape)

'''List of Preprocessing steps:                             
1) Retain only important fields 'Credit_History', 'Property_Area', 'Married' & 'LoanAmount'.      
2) Convert the field 'Married' into '0's & '1's using Label Encoder.              
3) Use one-hot encoder for the field 'Property_Area'.  
4) LoanAmount is rescaled using Standard Scaler
'''

train_x = train_file[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate',
                      'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',
                      'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',
                      'TotalWorkingYears', 'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	, 'YearsInCurrentRole',
                      'YearsSinceLastPromotion',	'YearsWithCurrManager', 'BusinessTravel', 'Department',
                      'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
train_y = train_file['Attrition']

# Replace the yes/no values with 0/1
train_y = train_y.replace({'Y': 1, 'N': 0})
train_y = train_y.replace({'Yes': 1, 'No': 0})

# Replace the yes/no values with 0/1
train_x = train_x.replace({'Y': 1, 'N': 0})
train_x = train_x.replace({'Yes': 1, 'No': 0})

# Display first 5 records from train_x
print(train_x.head())

# Display first 5 records from train_y
print(train_y.head())

print(train_x.head())

train_x.info()

numeric_features = ['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate',
                           'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',
                           'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',
                           'TotalWorkingYears', 'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	, 'YearsInCurrentRole',
                           'YearsSinceLastPromotion',	'YearsWithCurrManager']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_features = ['BusinessTravel', 'Department',
                        'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
categorical_transformer = OneHotEncoder(drop='first', sparse=False)

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)])


clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(random_state=100))])

clf.fit(train_x, train_y)


pickle.dump(clf, open('model.pkl', 'wb'))
