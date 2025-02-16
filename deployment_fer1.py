import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

df = pd.read_csv("crop_yield_data.csv")

print(df.head())

df.fillna(df.mean(), inplace=True)

le = LabelEncoder()
df['Crop_Type'] = le.fit_transform(df['Crop_Type'])

features = ['Rainfall', 'Temperature', 'Area_Harvested', 'Crop_Type']
target = 'Yield'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# XGBoost Model Training.. XGBRegressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} - MAE: {mae:.2f}, R² Score: {r2:.2f}")

evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("XGBoost", y_test, xgb_preds)

joblib.dump(rf_model, "crop_yield_model.pkl")
print("Model saved as crop_yield_model.pkl")

# Django API Endpoint... final term ok
@csrf_exempt
def predict_crop_yield(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            input_data = np.array([[data['Rainfall'], data['Temperature'], data['Area_Harvested'], le.transform([data['Crop_Type']])[0]]])
            input_data = scaler.transform(input_data)
            prediction = rf_model.predict(input_data)[0]
            return JsonResponse({"predicted_yield": prediction})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"message": "Send a POST request with input data."}, status=400)
