from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import io

app = Flask(__name__)

# Load and preprocess dataset
def preprocess_data(filepath):
    data = pd.read_csv(filepath, delimiter=';')
    data = data.drop(['DrugSerNo', 'Recipe'], axis=1)
    data['ReceteNetAmount'] = data['ReceteNetAmount'].str.replace(',', '.').astype(float)
    data['IlacReceteTutari'] = data['IlacReceteTutari'].str.replace(',', '.').astype(float)
    data['ReceteBrutTotal'] = data['ReceteBrutTotal'].str.replace(',', '.').astype(float)
    for col in ['Drug Dosage Expiry Date', 'Pickup Time', 'RecipeDate', 'Receipt Date']:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    for col in ['Drug Name', 'Field of Medicine']:
        data[col] = data[col].fillna('Unknown')
    label_encoders = {}
    categorical_columns = ['Drug Name', 'Pharmacy', 'Season', 'Drug Group', 'Field of Medicine']
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    data['Month'] = data['RecipeDate'].dt.month
    return data

# Route for uploading and processing the dataset
@app.route('/upload', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    data = preprocess_data(file)
    return jsonify({'message': 'File uploaded and processed successfully', 'rows': len(data)}), 200

# Route to get top-selling medicine types by month
@app.route('/top-medicines', methods=['GET'])
def top_medicines():
    filepath = request.args.get('filepath')
    if not filepath:
        return jsonify({'error': 'File path not provided'}), 400
    data = preprocess_data(filepath)
    monthly_sales = data.groupby(['Month', 'Drug Group'])['ReceteNetAmount'].sum().reset_index()
    top_medicines = monthly_sales.loc[monthly_sales.groupby('Month')['ReceteNetAmount'].idxmax()]
    results = top_medicines.to_dict(orient='records')
    return jsonify(results), 200

# Route for training and evaluating the model
@app.route('/train-model', methods=['GET'])
def train_model():
    filepath = request.args.get('filepath')
    if not filepath:
        return jsonify({'error': 'File path not provided'}), 400
    data = preprocess_data(filepath)
    X = data.drop('ReceteNetAmount', axis=1)
    y = data['ReceteNetAmount']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return jsonify({'Mean Squared Error': mse, 'R2 Score': r2}), 200

# Main
if __name__ == '__main__':
    app.run(debug=True)