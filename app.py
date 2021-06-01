import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)  # Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus',
                                                 'Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate',
                                                 'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',
                                                 'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',
                                                 'TotalWorkingYears', 'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	, 'YearsInCurrentRole',
                                                 'YearsSinceLastPromotion',	'YearsWithCurrManager'])

    #data_unseen = pd.DataFrame([final], columns = [ 'Credit_History', 'Property_Area', 'Married', 'LoanAmount'])

    prediction = model.predict(data_unseen)

    output = prediction

    if output == 1:
        label = 'Employee will leave'
    else:
        label = 'Employee will stay'

    return render_template('index.html', prediction_text='Emplyee Prediction: {}'.format(label))


if __name__ == "__main__":
    app.run(debug=True)
