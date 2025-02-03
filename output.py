import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all but error messages from TensorFlow
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import pandas as pd
import calendar
import sys
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

    model_path = 'hospital_bed_model_' + str(model_month) + '.keras'
    file_path = 'Aiims Jodhpur_' + str(model_month) + '.xlsx'

    # Load the trained model and scaler
    model = load_model(model_path)  # Using .keras format
    scaler = joblib.load('scaler.pkl')

    # Load the original data to get the feature columns
    df = pd.read_excel(file_path)
    df = df[pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce').notna()]
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # Scale the data
    scaled_data = scaler.transform(df)

    # Ensure that the requested date is in the future of the last available date
    last_date = df.index[-1]
    
    # Check if the input date is in the future of the last available date
    if date <= last_date:
        return None

    # Generate input sequence from the last available data
    last_sequence = scaled_data[-30:]
    predicted_beds = []
    prediction_dates = []

    while last_date < date:
        # Predict the next day
        prediction = model.predict(last_sequence[np.newaxis, :, :])
        predicted_beds.append(prediction[0])
        if last_date.day < calendar.monthrange(last_date.year, last_date.month)[1]:
            predicted_for_date = last_date + timedelta(days=1)
        else:
            predicted_for_date = datetime(last_date.year + 1, model_month, 1)
        prediction_dates.append(predicted_for_date)
        print(predicted_for_date)

        # Update last sequence with the prediction for the next step
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
        
        last_date = predicted_for_date


    # Scale the prediction back to the original values
    predicted_beds = scaler.inverse_transform(predicted_beds)
    wards = df.columns

    # Create a DataFrame for the predictions
    prediction_df = pd.DataFrame(predicted_beds, columns=wards, index=prediction_dates)
    prediction_df.index.name = 'Date'

    return prediction_df.loc[[date]]

    
# Input date from the user
st.set_page_config(page_title="Bed Occupency Predictor")
st.title("🛏️ Bed Occupency Predictor 🛏️")

# User input for DOB using a calendar date input
input_date = st.date_input(
    "Select Date :", 
    value=datetime.today(), 
    min_value=datetime(2022, 1, 1), 
    max_value=datetime.today()  # Restrict selection to today or earlier
)

if st.button("Predict Occupancy"):
    if input_date:
        df = predict_beds(input_date)
        df.loc['MAX'] = [500,300,100,30,30]
        df = df.apply(np.floor)
        st.write("Predicted Occupancy:")
        st.dataframe(df)
    else:
        st.error("Please select a Date.")

st.markdown("---")