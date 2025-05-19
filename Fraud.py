import pandas as pd
import numpy as np
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import plotly.express as px
import plotly.figure_factory as ff
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# 1. Data Acquisition & Preprocessing
def load_and_preprocess_data(file_path='C:\\zidio development\\p3\\creditcard.csv'):
    st.write("🔄 Loading and preprocessing data...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at {file_path}. Please ensure the file exists.")
        return None

    df = df.dropna()

    scaler = StandardScaler()
    df['Scaled_Amount'] = scaler.fit_transform(df[['Amount']])
    df['Scaled_Time'] = scaler.fit_transform(df[['Time']])
    df = df.drop(['Amount', 'Time'], axis=1)

    st.success("✅ Data preprocessing complete.")
    return df

# 2. Model Training
def train_models(X_train, y_train):
    models = {}

    with st.spinner("Training Logistic Regression..."):
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)
        models['Logistic Regression'] = lr

    with st.spinner("Training Random Forest..."):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf

    with st.spinner("Training XGBoost..."):
        xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb

    with st.spinner("Training Isolation Forest..."):
        iso_forest = IsolationForest(contamination=0.0017, random_state=42)
        iso_forest.fit(X_train)
        models['Isolation Forest'] = iso_forest

    st.success("✅ Model training complete.")
    return models

# 3. Export Predictions to CSV for Power BI
def export_to_powerbi_csv(df_full, model, model_name):
    df = df_full.copy()
    features = df.drop(['Class'], axis=1)

    if model_name == 'Isolation Forest':
        predictions = model.predict(features)
        predictions = np.where(predictions == -1, 1, 0)
    else:
        predictions = model.predict(features)

    if len(predictions) != len(df):
        raise ValueError("Prediction and DataFrame row counts do not match!")

    df['Prediction'] = predictions

    # Use timestamp to avoid file lock conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"fraud_predictions_powerbi_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    st.success(f"✅ Exported predictions to {output_path} for Power BI.")

# 4. Real-Time Monitoring Simulation
def simulate_real_time_monitoring(df, model, model_name):
    features = df.drop(['Class'], axis=1)

    if model_name == 'Isolation Forest':
        predictions = model.predict(features)
        predictions = np.where(predictions == -1, 1, 0)
    else:
        predictions = model.predict(features)

    df['Prediction'] = predictions
    return df

# 5. Streamlit Dashboard
def create_dashboard(df, models, X_test, y_test):
    st.title("💳 Financial Fraud Detection Dashboard")

    model_name = st.selectbox("🔍 Select a Model", list(models.keys()))

    st.subheader(f"📊 Evaluation Metrics for {model_name}")
    if model_name == 'Isolation Forest':
        predictions = models[model_name].predict(X_test)
        predictions = np.where(predictions == -1, 1, 0)
    else:
        predictions = models[model_name].predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    export_to_powerbi_csv(df.copy(), models[model_name], model_name)

    result_df = simulate_real_time_monitoring(df.copy(), models[model_name], model_name)

    st.subheader("🧮 Fraud vs Non-Fraud Distribution")
    fraud_counts = result_df['Prediction'].value_counts().reset_index()
    fraud_counts.columns = ['Prediction', 'Count']
    fraud_fig = px.bar(fraud_counts, x='Prediction', y='Count',
                       title='Fraud vs Non-Fraud Count',
                       labels={'Prediction': 'Class (0: Non-Fraud, 1: Fraud)'})
    st.plotly_chart(fraud_fig)

    st.subheader("📈 Anomaly / Fraud Score Distribution")
    columns_to_drop = ['Class']
    if 'Prediction' in df.columns:
        columns_to_drop.append('Prediction')
    df_for_scores = df.drop(columns_to_drop, axis=1)

    if model_name == 'Isolation Forest':
        scores = models[model_name].decision_function(df_for_scores)
        scores = -scores
    else:
        scores = models[model_name].predict_proba(df_for_scores)[:, 1]

    score_fig = px.histogram(x=scores, nbins=50, title='Anomaly/Fraud Score Histogram')
    st.plotly_chart(score_fig)

    st.subheader("🔗 Feature Correlation Heatmap")
    corr_matrix = df.drop(columns_to_drop, axis=1).corr()
    heatmap_fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        colorscale='Viridis',
        showscale=True
    )
    heatmap_fig.update_layout(title='Correlation Matrix', width=800, height=600)
    st.plotly_chart(heatmap_fig)

# 6. Main Function
def main():
    df = load_and_preprocess_data()
    if df is None:
        return

    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = train_models(X_train, y_train)
    create_dashboard(df, models, X_test, y_test)

if __name__ == '__main__':
    main()
