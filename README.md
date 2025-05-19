# Fraud detection_zidio
💳 Financial Fraud Detection Dashboard
This is a Streamlit-based web application that detects financial fraud using multiple machine learning models. It is built around the popular Kaggle credit card fraud detection dataset and supports real-time monitoring simulation, model evaluation, and export for Power BI integration.

🚀 Features
Load and preprocess credit card transaction data.

Train multiple models:

Logistic Regression

Random Forest

XGBoost

Isolation Forest (unsupervised)

Visual evaluation of each model’s performance.

Fraud vs Non-Fraud detection distribution.

Real-time prediction simulation.

Anomaly/fraud score histogram.

Feature correlation heatmap.

Export predictions to CSV (Power BI ready).

🛠️ Requirements
Install dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt should include:

text
Copy
Edit
pandas
numpy
streamlit
scikit-learn
xgboost
plotly
📦 How to Run
Ensure the dataset is in the correct path:

Place creditcard.csv in the path:

makefile
Copy
Edit
C:\zidio development\p3\
Or change the path in load_and_preprocess_data() inside Fraud.py.

Launch the Streamlit app:

bash
Copy
Edit
streamlit run Fraud.py
Interact with the dashboard to:

Choose a model.

View classification metrics.

Explore fraud detection visualizations.

Export predictions for Power BI.

📊 Output Example
fraud_predictions_powerbi_<timestamp>.csv is generated with model predictions for further analysis in Power BI or Excel.

📌 Notes
The application uses Stratified Train-Test Split to preserve class balance.

Isolation Forest is configured with a contamination rate of 0.0017.

Model performance and scores are visualized using Plotly charts and heatmaps.

📧 Author
Rishin Aiyappa A G
MCA - AI & ML Specialization
Jain (Deemed-to-be) University

