from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from prophet import Prophet
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset
file_path = 'pharmacy_sales_april_2023.csv'
sales_data = pd.read_csv(file_path)
sales_data['Time'] = pd.to_datetime(sales_data['Time'])
sales_data['Day'] = sales_data['Time'].dt.day
sales_data['Weekday'] = sales_data['Time'].dt.weekday
sales_data['TotalSales'] = sales_data['Price'] * sales_data['Quantity']

# Prepare data for RandomForest
features = sales_data[['DrugClass', 'Price', 'Quantity', 'Day', 'Weekday']]
target = sales_data['TotalSales']

encoder = OneHotEncoder(sparse_output=False)
encoded_classes = encoder.fit_transform(features[['DrugClass']])
encoded_df = pd.DataFrame(encoded_classes, columns=encoder.get_feature_names_out(['DrugClass']))
features = pd.concat([features.drop('DrugClass', axis=1).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Train RandomForest model
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Train Prophet model for time-series forecasting
sales_ts = sales_data.groupby(sales_data['Time'].dt.date)['TotalSales'].sum().reset_index()
sales_ts.columns = ['ds', 'y']
prophet_model = Prophet()
prophet_model.fit(sales_ts)
joblib.dump(prophet_model, 'prophet_model.pkl')

@app.route('/predict_regression', methods=['GET', 'POST'])
def predict_regression():
    if request.method == 'POST':
        data = request.json
        drug_class = data['DrugClass']
        price = data['Price']
        quantity = data['Quantity']
        day = data['Day']
        weekday = data['Weekday']

        # Load encoder and model
        encoder = joblib.load('encoder.pkl')
        encoded_class = encoder.transform([[drug_class]])
        encoded_df = pd.DataFrame(encoded_class, columns=encoder.get_feature_names_out(['DrugClass']))
        features = pd.DataFrame([[price, quantity, day, weekday]], columns=['Price', 'Quantity', 'Day', 'Weekday'])
        input_data = pd.concat([features, encoded_df], axis=1)

        model = joblib.load('random_forest_model.pkl')
        prediction = model.predict(input_data)
        return jsonify({'predicted_sales': prediction[0]})
    else:
        return jsonify({"error": "Only POST method is allowed"}), 405

@app.route('/predict_forecast', methods=['GET'])
def predict_forecast():
    future_periods = int(request.args.get('periods', 10))
    model = joblib.load('prophet_model.pkl')
    future = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)
    forecast_data = forecast[['ds', 'yhat']].tail(future_periods).to_dict(orient='records')
    return jsonify(forecast_data)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
