# Streamlit app to let users get housing prices with a given input, based on a model artifact

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
@st.cache_resource
def load_model(path):
    print("model loaded")
    return joblib.load(path)

model = load_model('/Users/reymerekar/Desktop/ml_pipeline_airflow/artifacts/model.joblib')

# Set up the title and input fields as before
st.title('Boston Housing Price Prediction')

# Creating user input forms
st.subheader('Please enter the housing attributes:')

# Assuming the model expects features like CRIM, ZN, etc. Adjust according to your features.
CRIM = st.number_input('Per capita crime rate by town (CRIM)')
ZN = st.number_input('Proportion of residential land zoned for lots over 25,000 sq.ft. (ZN)')
INDUS = st.number_input('Proportion of non-retail business acres per town (INDUS)')
CHAS = st.selectbox('Charles River dummy variable (CHAS)', [0, 1])
NOX = st.number_input('Nitric oxides concentration (parts per 10 million) (NOX)')
RM = st.number_input('Average number of rooms per dwelling (RM)')
AGE = st.number_input('Proportion of owner-occupied units built prior to 1940 (AGE)')
DIS = st.number_input('Weighted distances to five Boston employment centres (DIS)')
RAD = st.number_input('Index of accessibility to radial highways (RAD)')
TAX = st.number_input('Full-value property tax rate per $10,000 (TAX)')
PTRATIO = st.number_input('Pupil-teacher ratio by town (PTRATIO)')
B = st.number_input('1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town (B)')
LSTAT = st.number_input('Percentage lower status of the population (LSTAT)')


# Button to make predictions
if st.button('Predict Housing Prices'):
    # Create a DataFrame based on the inputs
    input_data = pd.DataFrame([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
                              columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
    
    # Convert the DataFrame to a numpy array
    input_array = input_data.to_numpy()

    # Get the prediction
    prediction = model.predict(input_array)
    
    # Display the prediction
    st.subheader(f'The predicted housing price is: ${prediction[0]*1000:.2f}')

if st.button('Load Example and Predict'):
    # Assuming the sample payload is stored in 'sample_payload.csv'
    sample_data = pd.read_csv('/Users/reymerekar/Desktop/ml_pipeline_airflow/data/sample_payload.csv', index_col = 0)
    sample_array = sample_data.to_numpy()
    prediction = model.predict(sample_array)
    st.subheader(f'Prediction from example payload: ${prediction[0]*1000:.2f}')

