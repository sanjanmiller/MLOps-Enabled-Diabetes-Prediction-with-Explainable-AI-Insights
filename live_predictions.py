import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

# --- Load trained model and scaler ---
model = joblib.load(r'C:\Users\HP\random_forest_diabetes_model.pkl')
scaler = joblib.load(r'C:\Users\HP\scaler.pkl')

# --- Load live dataset ---
live_df = pd.read_csv(r'C:\Users\HP\Downloads\live_diabetes_data.csv')  # should have same features as training
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Scale live data
X_live_scaled = scaler.transform(live_df[features])

# --- Run predictions ---
predictions = model.predict(X_live_scaled)

# --- Compute metrics ---
total_patients = len(predictions)
diabetic_count = int(np.sum(predictions == 1))
non_diabetic_count = int(np.sum(predictions == 0))

print(f"Total Patients: {total_patients}")
print(f"Predicted Diabetic: {diabetic_count}")
print(f"Predicted Non-Diabetic: {non_diabetic_count}")

# --- Log everything to MLflow ---
mlflow.set_experiment("Diabetes_Prediction_Live")

with mlflow.start_run():
    # Parameters
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("max_depth", model.max_depth)

    # Metrics
    mlflow.log_metric("total_patients", total_patients)
    mlflow.log_metric("predicted_diabetic", diabetic_count)
    mlflow.log_metric("predicted_non_diabetic", non_diabetic_count)

    # Artifacts
    mlflow.sklearn.log_model(model, "random_forest_model")
    mlflow.log_artifact(r'C:\Users\HP\scaler.pkl')  # log scaler
    mlflow.log_artifact(r'C:\Users\HP\VS CODE\shap_feature_importance_bar.png')  # log existing SHAP plot

print("Live predictions and SHAP plot successfully logged to MLflow!")
