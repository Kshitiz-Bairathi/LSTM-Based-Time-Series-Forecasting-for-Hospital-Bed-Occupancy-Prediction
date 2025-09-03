import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import pandas as pd
import calendar
import streamlit as st

def create_sequences(data, sequence_length=30):
    X = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
    return np.array(X)

# Function to make predictions and display results in table format
def predict_beds(date_str):
    date = pd.Timestamp(date_str)   
    model_month = date.month

    model_path = f"hospital_bed_model_{model_month}.keras"
    file_path = f"Aiims Jodhpur_{model_month}.xlsx"

    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found for {calendar.month_name[model_month]} ({model_path}).")
        return None
    
    # Check if data file exists
    if not os.path.exists(file_path):
        st.error(f"❌ Data file not found for {calendar.month_name[model_month]} ({file_path}).")
        return None

    # Load the trained model and scaler
    model = load_model(model_path)
    scaler = joblib.load("scaler.pkl")
    
    # Load and preprocess data
    df = pd.read_excel(file_path)
    df = df[pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce').notna()]
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    scaled_data = scaler.transform(df)
    last_date = df.index[-1]
    
    if date <= last_date:
        return None

    # Prediction loop
    last_sequence = scaled_data[-30:]
    predicted_beds, prediction_dates = [], []

    while last_date < date:
        prediction = model.predict(last_sequence[np.newaxis, :, :])
        predicted_beds.append(prediction[0])
        
        if last_date.day < calendar.monthrange(last_date.year, last_date.month)[1]:
            predicted_for_date = last_date + timedelta(days=1)
        else:
            predicted_for_date = datetime(last_date.year + 1, model_month, 1)
        
        prediction_dates.append(predicted_for_date)
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
        last_date = predicted_for_date

    predicted_beds = scaler.inverse_transform(predicted_beds)
    wards = df.columns

    prediction_df = pd.DataFrame(predicted_beds, columns=wards, index=prediction_dates)
    prediction_df.index.name = "Date"

    return prediction_df.loc[[date]]


# Streamlit App
st.set_page_config(page_title="Bed Occupancy Predictor")
st.title("🛏️ Bed Occupancy Predictor 🛏️")

allowed_months = {
    1: "January", 
    2: "February", 
    3: "March", 
    4: "April", 
    11: "November", 
    12: "December"
}

# User selects year and month separately
year = st.number_input("Select Year:", min_value=2022, max_value=datetime.today().year, value=datetime.today().year, step=1)
month = st.selectbox("Select Month:", options=list(allowed_months.keys()), format_func=lambda x: allowed_months[x])
day = st.number_input("Select Day:", min_value=1, max_value=31, value=datetime.today().day, step=1)

try:
    input_date = datetime(year, month, day)
except ValueError:
    st.error("❌ Invalid day for the selected month. Please adjust the date.")
    st.stop()

if st.button("Predict Occupancy"):
    df = predict_beds(input_date)
    if df is not None:
        df.loc['MAX'] = [500, 300, 100, 30, 30]
        df = df.apply(np.floor)
        st.write("Predicted Occupancy:")
        st.dataframe(df)
    else:
        st.error("Selected date must be after the last available data date.")

st.markdown("---")
