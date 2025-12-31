import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
# We use @st.cache_resource so it only loads once, making the app faster
@st.cache_resource
def load_model():
    return joblib.load('sepsis_model_v1.pkl')

model = load_model()

# 2. App Title and Description
st.title("🏥 Sepsis Risk Prediction System")
st.markdown("""
This AI tool helps clinicians assess the risk of sepsis in ICU patients. 
Enter the vital signs below to generate a risk score.
*Model trained on PhysioNet/MIMIC data patterns.*
""")

# 3. Create the Input Form
st.sidebar.header("Patient Vitals")

def user_input_features():
    # We use sliders and number inputs for easy data entry
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=65)
    hr = st.sidebar.slider("Heart Rate (bpm)", 40, 200, 85)
    bp = st.sidebar.slider("Systolic BP (mmHg)", 40, 200, 120)
    temp = st.sidebar.number_input("Temperature (°C)", 34.0, 42.0, 37.0, step=0.1)
    wbc = st.sidebar.number_input("WBC Count (10^9/L)", 0.0, 50.0, 10.0, step=0.1)
    
    # Store in a dictionary
    data = {
        'Age': age,
        'HeartRate': hr,
        'SysBP': bp,
        'Temp': temp,
        'WBC': wbc
    }
    return pd.DataFrame([data])

# Get input from user
input_df = user_input_features()

# 4. Display Patient Data
st.subheader("Patient Summary")
st.write(input_df)

# 5. Prediction Logic
if st.button("Analyze Risk"):
    # Calculate Shock Index on the fly
    input_df['ShockIndex'] = input_df['HeartRate'] / input_df['SysBP']
    
    # Get Probability
    prediction_prob = model.predict_proba(input_df)[0][1]
    
    # Display Result
    st.subheader("Clinical Assessment")
    
    # Visual Progress Bar for Risk
    st.progress(prediction_prob)
    
    if prediction_prob > 0.8:
        st.error(f"🔴 CRITICAL RISK DETECTED (Score: {prediction_prob:.2%})")
        st.write("**Recommendation:** Initiate Sepsis Protocol immediately. Call Rapid Response.")
    elif prediction_prob > 0.5:
        st.warning(f"⚠️ HIGH RISK (Score: {prediction_prob:.2%})")
        st.write("**Recommendation:** Monitor lactate and vitals hourly.")
    else:
        st.success(f"🟢 LOW RISK (Score: {prediction_prob:.2%})")
        st.write("**Recommendation:** Standard observation.")
        
    st.caption(f"Calculated Shock Index: {input_df['ShockIndex'].values[0]:.2f} (Normal range: 0.5 - 0.7)")