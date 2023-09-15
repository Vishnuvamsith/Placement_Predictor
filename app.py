
from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the pre-trained model
path=os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(path,'placement_prediction_model .pkl'))

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():


    # Get input values from user
    Internships = int(request.form['Internships'])
    CGPA = int(request.form['CGPA'])
    Hostel= int(request.form['Hostel'])
    HistoryOfBacklogs = int(request.form['HistoryOfBacklogs'])

    # Create input data array
    input_data = np.array([[Internships,CGPA,Hostel,HistoryOfBacklogs]])

    # Predict the price using the pre-trained model
    predicted_placement = model.predict(input_data)[0]
    if predicted_placement==1:
        val="You will be placed"
    else:
        val="You will not be placed"
    # Render the result page with predicted price
    return render_template('index.html', val=val)

if __name__ == '__main__':
    app.run(debug=True)