# 🧑‍💼 Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to leave the company based on key HR and performance metrics. Deployed with an interactive **Streamlit** web interface.

## 🚀 Features

- Predicts employee attrition: **Yes** or **No**
- Built using algorithms like **Random Forest**, **Logistic Regression**, and **XGBoost**
- Clean, interactive frontend using **Streamlit**
- Models serialized using `joblib` for fast loading
- Visualizes input features and prediction output clearly

## 🧠 ML Models Used

- Logistic Regression
- Random Forest Classifier 🌲
- XGBoost Classifier ⚡

## 📁 Project Structure

```
employee_attrition/
├── attrition_app.py # Streamlit frontend app
├── attrition_model.pkl # Trained ML model
├── label_encoder.pkl # Encoder for categorical labels
├── requirements.txt # Required Python libraries
├── README.md # Project documentation
└── data/
└── employee_data.csv # Training dataset
```
