import streamlit as st
import joblib
import pandas as pd
from sklearn.exceptions import NotFittedError
import numpy as np



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



def main():

    st.title("Maternal Health Risk Prediction App")
    st.write("This app predicts the risk level of maternal health based on user input features.")


# Create tabs for different app sections
    tab1, tab2, tab3, tab4 = st.tabs(["About", "EDA", "Prediction", "Contact"])
    with tab1:
        st.header("About")
        st.write("""
            This application uses a Random Forest model to predict the risk level of maternal health based on various input features.
            The model has been trained on a dataset containing features such as age, blood pressure, blood sugar, body temperature, and heart rate.
            The risk levels are categorized as:
            - Low risk (2)
            - Moderate risk (1)
            - High risk (0)
                 
    
        """)
        st.write("""
            The app allows users to input patient details and receive a risk prediction.
            It also includes sections for exploratory data analysis (EDA) and contact information.
        """)
        st.write(""" ### Model Performance
        
        The machine learning model was trained on historical patient data with known outcomes. The model achieves:
        
        - Accuracy: ~87%
        - Precision: ~96% 
        - Recall: ~95%
        - ROC-AUC: ~95%
        
        These metrics indicate good but not perfect predictive ability. Always consult healthcare professionals for 
        medical decisions.
        """)
        st.write("""The model is designed to assist healthcare professionals in making informed decisions about maternal health.
            It is not a substitute for professional medical advice, diagnosis, or treatment.
            Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
        """)

    with tab2:
        st.header("Exploratory Data Analysis (EDA)")
        st.write("""
            In this section, you can explore the dataset used for training the model.
            You can visualize the distribution of features and their relationships with the target variable.
            However, for simplicity, we will not include detailed EDA in this app.
        """)
        st.write("""
            The dataset contains the following features:
            - Age: Age of the patient
            - SystolicBP: Systolic Blood Pressure
            - DiastolicBP: Diastolic Blood Pressure
            - BS: Blood Sugar level
            - BodyTemp: Body Temperature
            - HeartRate: Heart Rate
            - RiskLevel: Target variable indicating risk level (0, 1, 2)
        """)
        st.write(""" You can visualize the data using various plots, such as histograms, scatter plots, and box plots.
            However, for this app, we will focus on the prediction functionality.
        """)
        # Display exploratory data analysis visualizations
        st.markdown("<div class='sub-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
        st.markdown("""This section provides visualizations to help understand the relationships between different patient characteristics and survival outcomes. These insights can help interpret the model predictions.
                    """)
        
        # Generate sample data for visualizations
        import random
        sample_size = 1000  # Define the sample size for the EDA
        data = {
            'Age': [random.randint(10, 85) for _ in range(sample_size)],
            'SystolicBP': [random.randint(60, 200) for _ in range(sample_size)],
            'DiastolicBP': [random.randint(40, 120) for _ in range(sample_size)],
            'BS': [random.randint(4, 20) for _ in range(sample_size)],
            'BodyTemp': [random.randint(95, 105) for _ in range(sample_size)],
            'HeartRate': [random.randint(70, 100) for _ in range(sample_size)],
            'RiskLevel': [random.choice([0, 1, 2]) for _ in range(sample_size)]
        }
        df = pd.DataFrame(data)
        st.subheader("Sample Data")
        st.write(df)
        st.subheader("Feature Distributions")
        st.bar_chart(df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']].mean())
        st.subheader("Risk Level Distribution")
        risk_counts = df['RiskLevel'].value_counts()
        st.bar_chart(risk_counts)

        # Create create a sample scatter plot
        st.subheader("Scatter Plot of Age vs Systolic Blood Pressure")
        st.write("""            This scatter plot shows the relationship between Age and Systolic Blood Pressure.
            It can help identify trends or patterns in the data.
        """)
        st.line_chart(df[['Age', 'SystolicBP']].set_index('Age'))   

        # Create hitmap for feature correlation
        st.subheader("Feature Correlation Heatmap")
        st.write("""            This heatmap shows the correlation between different features in the dataset.
            It can help identify which features are strongly correlated with each other.
        """)
        st.write(df.corr())
        st.write(""" The EDA section provides a basic overview of the dataset and its features.
            For more detailed analysis, you can explore the dataset further using various visualization libraries.
        """)    

    with tab3:
        st.header("Prediction")
        st.write("""
            In this section, you can input patient details to predict the risk level.
            The model will output whether the patient is at low, moderate, or high risk based on the input features.
        """)
        # Input fields for user data
        # Display the user input form
        st.header("Enter Patient Details")
    
        # Input widgets for numerical features
        age = st.number_input("Enter Age:", min_value=10, max_value=70, value=30)
        systolic_bp = st.number_input("Enter Systolic Blood Pressure:", min_value=70, max_value=200, value=120)
        diastolic_bp = st.number_input("Enter Diastolic Blood Pressure:", min_value=45, max_value=130, value=80)
        bs = st.number_input("Enter Blood Sugar (BS):", min_value=2.0, max_value=20.0, value=5.5, format="%.2f")
        body_temp = st.number_input("Enter Body Temperature:", min_value=95.0, max_value=105.0, value=98.0, format="%.1f")
        heart_rate = st.number_input("Enter Heart Rate:", min_value=40, max_value=180, value=75)


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

        if predict_button:
            # Check if the model is loaded
            if model is not None and scaler is not None and label_encoder is not None:
                # Make predictions
                user_input_scaled = scaler.transform(user_input_df)
                prediction = model.predict(user_input_scaled)

                # Decode the prediction
                if isinstance(prediction, (list, pd.Series)):
                    prediction = prediction[0]
                elif isinstance(prediction, (np.ndarray, pd.DataFrame)):
                    prediction = prediction.item()
                decoded_prediction = label_encoder.inverse_transform([prediction])[0]

                # Display the prediction
                st.subheader("Predicted Risk Level:")
                if decoded_prediction == 0:
                    st.error("High risk - immediate medical attention required")
                elif decoded_prediction == 1:
                    st.warning("Moderate risk - monitor closely")
                elif decoded_prediction == 2:
                    st.success("Low risk - no immediate action required")
                else:
                    st.error("Error: Unexpected prediction output")
            else:
                st.error("Model is not loaded properly.")
                st.error("Please ensure the model and preprocessing objects are loaded correctly.")
    with tab4:
        st.header("Contact")
        st.write("""
            For any queries or feedback, please contact:
            - Email: support@maternhealthapp.com
            - Phone: +250-783-656-700
            - GitHub: [Maternal Health App](https://github.com/ndamukunda139/maternal_health_app)
        """)
    
    
if __name__ == "__main__":
    main()