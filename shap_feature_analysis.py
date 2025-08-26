import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset, scaler, model
df = pd.read_csv(r'C:\Users\HP\Downloads\archive\diabetes.csv')
scaler = joblib.load(r'C:\Users\HP\scaler.pkl')
model = joblib.load(r'C:\Users\HP\random_forest_diabetes_model.pkl')

# Features
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
X_scaled = scaler.transform(df[features])

# SHAP explainer and values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)[1]  # class 1 (diabetic)

# Compute mean absolute SHAP values
mean_shap = np.abs(shap_values).mean(axis=0)

# Print feature importance
print("\nFeature Importance (Higher = More Impact on Diabetes Prediction):\n")
for feature, importance in zip(features, mean_shap):
    print(f"{feature}: {importance:.4f}")

# Sort features for plotting
sorted_idx = np.argsort(mean_shap)[::-1]
features_sorted = [features[i] for i in sorted_idx]
importance_sorted = [mean_shap[i] for i in sorted_idx]

# --- Horizontal bar chart ---
plt.figure(figsize=(10, 6))

# Bar plot
plt.barh(features_sorted, importance_sorted, color='skyblue')

# Labels
plt.xlabel("Impact on Diabetes Prediction", fontsize=12)
plt.ylabel("Patient Health Features", fontsize=12)
plt.title("Most Important Factors Affecting Diabetes Risk", fontsize=16, weight='bold')

# Highest importance on top
plt.gca().invert_yaxis()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()

# Save and show
plt.savefig('shap_feature_importance_bar.png')
plt.show()

