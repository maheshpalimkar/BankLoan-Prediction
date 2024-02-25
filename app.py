# app.py

from flask import Flask, render_template, request, jsonify
from utils import PredictModel
import config

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs from the form
        age = float(request.form['age'])
        experience = float(request.form['experience'])
        income = float(request.form['income'])
        family = int(request.form['family'])
        cc_avg = float(request.form['cc_avg'])
        education = int(request.form['education'])
        mortgage = float(request.form['mortgage'])
        securities_account = int(request.form['securities_account'])
        cd_account = int(request.form['cd_account'])
        online = int(request.form['online'])
        credit_card = int(request.form['credit_card'])

        model_predictor = PredictModel()
        # Create a test array without 'ID' and 'ZIP.Code'
        test_array = model_predictor.create_test_array(age, experience, income, family, cc_avg, education, mortgage, securities_account, cd_account, online, credit_card)

        predicted_class = model_predictor.predict_class(test_array)
        predicted_class = int(predicted_class)

        # Render the result on a new page
        return jsonify({'predicted_class': predicted_class})
    
if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = config.PORT_NUMBER, debug=False)
