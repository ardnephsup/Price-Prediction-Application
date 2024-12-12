from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load saved models
models = {
    'LinearRegression': pickle.load(open('LinearRegression.pkl', 'rb')),
    'Ridge': pickle.load(open('Ridge.pkl', 'rb')),
    'Lasso': pickle.load(open('Lasso.pkl', 'rb')),
    'RandomForest': pickle.load(open('RandomForest.pkl', 'rb')),
    'SVR': pickle.load(open('SVR.pkl', 'rb')),
    'MLPRegressor': pickle.load(open('MLPRegressor.pkl', 'rb')),
    'XGBoost': pickle.load(open('XGBoost.pkl', 'rb')),
    'LightGBM': pickle.load(open('LightGBM.pkl', 'rb'))
}

# Load model evaluation results
evaluation_results = pd.read_csv(r"C:\Users\HP\OneDrive\naresh it DS\projects\my projectss\frontend,backend project for house price estimation\model_evaluation_results.csv")

@app.route('/')
def home():
    # Convert evaluation results to a list of dictionaries for easy display
    model_performance = evaluation_results.to_dict(orient='records')
    return render_template('index.html', model_performance=model_performance)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input values
        avg_income = float(request.form['avg_income'])
        house_age = float(request.form['house_age'])
        num_rooms = float(request.form['num_rooms'])
        num_bedrooms = float(request.form['num_bedrooms'])
        population = float(request.form['population'])
        model_name = request.form['model']
        
        # Prepare data for prediction
        features = np.array([[avg_income, house_age, num_rooms, num_bedrooms, population]])
        
        # Predict using the selected model
        model = models[model_name]
        prediction = model.predict(features)
        
        return render_template('index.html', model_performance=model_performance)


if __name__ == '__main__':
    app.run(debug=True)
