from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from prophet import Prophet
import joblib
import os

app = Flask(__name__)

file_path = 'pharmacy_sales_april_2023.csv'
sales_data = pd.read_csv(file_path)
sales_data['Time'] = pd.to_datetime(sales_data['Time'])
sales_data['Day'] = sales_data['Time'].dt.day
sales_data['Weekday'] = sales_data['Time'].dt.weekday
sales_data['TotalSales'] = sales_data['Price'] * sales_data['Quantity']
features = sales_data[['DrugClass', 'Price', 'Quantity', 'Day', 'Weekday']]
target = sales_data['TotalSales']

encoder = OneHotEncoder(sparse_output=False)
encoded_classes = encoder.fit_transform(features[['DrugClass']])
encoded_df = pd.DataFrame(encoded_classes, columns=encoder.get_feature_names_out(['DrugClass']))
features = pd.concat([features.drop('DrugClass', axis=1).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(encoder, 'encoder.pkl')
drug_classes = sales_data['DrugClass'].unique()
os.makedirs('prophet_models', exist_ok=True)

for drug_class in drug_classes:
    class_data = sales_data[sales_data['DrugClass'] == drug_class]
    sales_ts = class_data.groupby(class_data['Time'].dt.date)['TotalSales'].sum().reset_index()
    sales_ts.columns = ['ds', 'y']    
    prophet_model = Prophet()
    prophet_model.fit(sales_ts)    
    model_path = f'prophet_models/prophet_{drug_class}.pkl'
    joblib.dump(prophet_model, model_path)

@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    data = request.json
    drug_class = data['DrugClass']
    price = data['Price']
    quantity = data['Quantity']
    day = data['Day']
    weekday = data['Weekday']

    encoder = joblib.load('encoder.pkl')
    encoded_class = encoder.transform([[drug_class]])
    encoded_df = pd.DataFrame(encoded_class, columns=encoder.get_feature_names_out(['DrugClass']))
    features = pd.DataFrame([[price, quantity, day, weekday]], columns=['Price', 'Quantity', 'Day', 'Weekday'])
    input_data = pd.concat([features, encoded_df], axis=1)
    model = joblib.load('random_forest_model.pkl')
    prediction = model.predict(input_data)
    return jsonify({'predicted_sales': prediction[0]})

# Route for Prophet forecast by DrugClass
@app.route('/predict_forecast', methods=['GET'])
def predict_forecast():
    drug_class = request.args.get('DrugClass')
    future_periods = int(request.args.get('periods', 10))    
    model_path = f'prophet_models/prophet_{drug_class}.pkl'
    
    if not os.path.exists(model_path):
        return jsonify({'error': f"No forecasting model found for DrugClass '{drug_class}'"}), 404    
    model = joblib.load(model_path)
    future = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)
    forecast_data = forecast[['ds', 'yhat']].tail(future_periods).to_dict(orient='records')
    return jsonify({drug_class: forecast_data})

if __name__ == '__main__':
    app.run(debug=True)
