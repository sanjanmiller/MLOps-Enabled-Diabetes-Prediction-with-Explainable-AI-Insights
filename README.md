# 🩺 MLOps-Enabled Diabetes Prediction with Explainable AI Insights  

This project is an **End-to-End Machine Learning system** for predicting diabetes risk. It integrates:  
- **Random Forest model** for classification  
- **MLOps with MLflow** for experiment tracking & live monitoring  
- **Explainable AI (XAI) with SHAP** for transparent feature importance  

---

## 📌 Features  
✔️ **Random Forest-based classifier** trained on the Pima Indians Diabetes dataset.  
✔️ **Data preprocessing pipeline** (handling missing values, scaling).  
✔️ **Global explainability with SHAP** — bar plot showing top risk factors for diabetes.  
✔️ **MLOps integration with MLflow** — logs metrics, predictions, and artifacts.  
✔️ **Live prediction support** for batch or streaming patient data.  

---

## 🖥️ Demo Output  

**🔹 SHAP Global Feature Importance**  
<img src="https://raw.githubusercontent.com/sanjanmiller/MLOps-Enabled-Diabetes-Prediction-with-Explainable-AI-Insights/refs/heads/main/shap_feature_importance_bar.png" width="600">  

**🔹 MLflow Experiment Tracking**  
<img src="https://raw.githubusercontent.com/sanjanmiller/MLOps-Enabled-Diabetes-Prediction-with-Explainable-AI-Insights/refs/heads/main/diabetes1.JPG" width="600">  
<img src="https://raw.githubusercontent.com/sanjanmiller/MLOps-Enabled-Diabetes-Prediction-with-Explainable-AI-Insights/refs/heads/main/diabetes.JPG" width="600"> 

---

## 🔧 **Installation & Setup**  

### **1️⃣ Clone the Repository**  
git clone https://github.com/sanjanmiller/MLOps-Enabled-Diabetes-Prediction-with-Explainable-AI-Insights.git  
cd MLOps-Enabled-Diabetes-Prediction-with-Explainable-AI-Insights  

### **2️⃣ Install Dependencies**  
pip install -r requirements.txt  

### **3️⃣ Run the Project**  
python live_predictions.py 

### **4️⃣ Run MLflow Tracking Server**
mlflow ui 

