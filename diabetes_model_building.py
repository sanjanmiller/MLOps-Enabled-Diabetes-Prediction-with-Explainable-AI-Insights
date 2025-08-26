import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Load the dataset
df = pd.read_csv(r'C:\Users\HP\Downloads\archive\diabetes.csv')

# Replace zeros with NaN and fill with the mean of the column
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=10,
    min_samples_leaf=1,
    min_samples_split=2,
    criterion='gini',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest Classifier: {accuracy:.4f}")

# --- Save the trained model ---
joblib.dump(model, 'random_forest_diabetes_model.pkl')
print("Model saved as 'random_forest_diabetes_model.pkl'")

# ---save the scaler ---
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")

