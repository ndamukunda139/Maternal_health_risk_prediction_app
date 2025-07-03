import streamlit as st
import joblib
import pandas as pd
from sklearn.exceptions import NotFittedError



# Load the saved model, scaler, and label encoder
try:
    model = joblib.load('default_random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    st.success("Model and preprocessing objects loaded successfully.")
except FileNotFoundError:
    st.error("Error: Model or preprocessing files not found. Please ensure 'default_random_forest_model.joblib', 'scaler.joblib', and 'label_encoder.joblib' are in the same directory.")
except Exception as e:
    st.error(f"An error occurred while loading the files: {e}")

st.title("Maternal Health Risk Prediction")

# Add content to the Streamlit app here in subsequent steps
# For now, we just confirm loading
st.write("Application is ready to predict maternal health risk.")



### Design the user interface


st.header("Enter Patient Details")


# Input widgets for numerical features
# Example hardcoded values (adjust as appropriate)
age = st.number_input("Enter Age:", min_value=15, max_value=50, value=30)
systolic_bp = st.number_input("Enter Systolic Blood Pressure:", min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input("Enter Diastolic Blood Pressure:", min_value=50, max_value=130, value=80)
bs = st.number_input("Enter Blood Sugar (BS):", min_value=2.0, max_value=20.0, value=5.5, format="%.2f")
body_temp = st.number_input("Enter Body Temperature:", min_value=95.0, max_value=105.0, value=98.0, format="%.1f")
heart_rate = st.number_input("Enter Heart Rate:", min_value=40, max_value=180, value=75)
# ...rest of your code...# You can add a button to trigger prediction in the next step
# predict_button = st.button("Predict Risk Level")


### Process user input

# Create a dictionary to hold the input features
user_input = {
    'Age': age,
    'SystolicBP': systolic_bp,
    'DiastolicBP': diastolic_bp,
    'BS': bs,
    'BodyTemp': body_temp,
    'HeartRate': heart_rate
}

# Convert the dictionary to a DataFrame
user_input_df = pd.DataFrame([user_input])

st.subheader("Input Data:")
st.write(user_input_df)

# Add a button to trigger prediction
predict_button = st.button("Predict Risk Level")

# Check if model is fitted

if predict_button:
    try:
        user_input_scaled = scaler.transform(user_input_df)
        prediction = model.predict(user_input_scaled)
        decoded_prediction = label_encoder.inverse_transform(prediction)[0]
        # ... display code as before ...
    except NotFittedError:
        st.error("One of the model components (scaler/model/encoder) is not fitted. Please retrain and save the components properly.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


### Preprocess user input

# Check if the predict button is clicked
if predict_button:
    # Apply the loaded scaler to the user input DataFrame
    user_input_scaled = scaler.transform(user_input_df)

    # Now, user_input_scaled contains the scaled features, ready for prediction
    # The prediction logic will be added in the next step

    # Use the loaded model to make a prediction
    prediction = model.predict(user_input_scaled)

    # The prediction result (numerical label 0, 1, or 2) is now stored in the 'prediction' variable
    # The next step will be to display the prediction and potentially the decoded risk level


### Display predictions

 # Decode the numerical prediction back to the original risk level string
    decoded_prediction = label_encoder.inverse_transform(prediction)[0]

    st.subheader("Predicted Risk Level:")

    # Display the decoded risk level with appropriate styling
    if decoded_prediction == 'low risk':
        st.success(decoded_prediction)
    elif decoded_prediction == 'mid risk':
        st.warning(decoded_prediction)
    elif decoded_prediction == 'high risk':
        st.error(decoded_prediction)
    else:
        st.write(decoded_prediction) # Fallback for unexpected output


def main():
    """Main function to run the Streamlit app"""
    st.markdown("<h1 class='main-header'>Maternal health risk Predictions</h1>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()