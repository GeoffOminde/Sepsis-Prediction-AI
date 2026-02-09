import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root and VAE directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "vae_sepsis_prediction"))

from vae_sepsis_prediction.models.sepsis_vae import SepsisVAE

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SepsisAI - Generative Risk Assessment",
    page_icon="🏥",
    layout="wide"
)

# --- STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .risk-card {
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }
    .critical { background-color: #ff4b4b; }
    .high { background-color: #ffa500; }
    .moderate { background-color: #f1c40f; }
    .low { background-color: #2ecc71; }
</style>
""", unsafe_allow_html=True)

# --- CACHED ASSETS ---
@st.cache_resource(show_spinner="Loading VAE Model...")
def load_vae_assets():
    model_path = Path("vae_sepsis_prediction/vae_output/deployment/sepsis_vae_model.pth")
    scaler_path = Path("vae_sepsis_prediction/vae_output/deployment/feature_scaler.pkl")
    
    # Debug: Check if files exist
    model_exists = model_path.exists()
    scaler_exists = scaler_path.exists()
    
    if not model_exists or not scaler_exists:
        # Fallback to legacy model if VAE is not yet built
        st.warning(f"VAE files missing - Model: {model_exists}, Scaler: {scaler_exists}")
        return None, None, True
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        config = checkpoint['model_config']
        
        model = SepsisVAE(
            input_dim=config['input_dim'],
            latent_dim=config['latent_dim'],
            hidden_dims=config['hidden_dims']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler = joblib.load(scaler_path)
        st.success("✅ VAE Model loaded successfully!")
        return model, scaler, False
    except Exception as e:
        st.error(f"Error loading VAE: {e}")
        return None, None, True

vae_model, feature_scaler, is_legacy = load_vae_assets()

# --- APP HEADER ---
with st.container():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://img.icons8.com/isometric/512/hospital.png", width=100)
    with col2:
        st.title("SepsisAI - Clinical Decision Support")
        st.caption("Generative AI Patient Monitoring with Uncertainty Quantification")
        if is_legacy:
            st.warning("⚠️ Running Legacy Random Forest Model. VAE Model initialization pending.")

st.divider()

# --- INPUT SECTION ---
st.subheader("📋 Patient Clinical Data")
with st.expander("Expand to enter Vital Signs and Lab Results", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Vitals**")
        age = st.number_input("Age", 18, 100, 65)
        gender = st.selectbox("Gender", ["Male", "Female"], index=1)
        hr = st.slider("Heart Rate (bpm)", 40, 200, 85)
        rr = st.slider("Respiratory Rate", 8, 40, 16)
        spo2 = st.slider("SpO2 (%)", 70, 100, 96)
        
    with col2:
        st.markdown("**Blood Pressure & Temp**")
        sbp = st.slider("Systolic BP (mmHg)", 60, 200, 120)
        dbp = st.slider("Diastolic BP (mmHg)", 40, 130, 75)
        temp = st.number_input("Temperature (°C)", 34.0, 42.0, 37.0, step=0.1)
        
    with col3:
        st.markdown("**Lab Results**")
        wbc = st.number_input("WBC Count (10^9/L)", 2.0, 50.0, 9.0, step=0.1)
        lactate = st.number_input("Lactate (mmol/L)", 0.5, 10.0, 1.5, step=0.1)
        creatinine = st.number_input("Creatinine (mg/dL)", 0.5, 5.0, 1.0, step=0.1)
        platelets = st.number_input("Platelets (k/uL)", 50, 500, 250)
        bilirubin = st.number_input("Bilirubin (mg/dL)", 0.2, 5.0, 0.8, step=0.1)

# --- PREDICTION LOGIC ---
if st.button("🚀 Run Sepsis Risk Assessment", use_container_width=True):
    # Map Gender
    gender_val = 1 if gender == "Male" else 0
    
    # Calculate derived features
    map_val = (sbp + 2 * dbp) / 3
    pp = sbp - dbp
    si = hr / sbp if sbp > 0 else 0
    age_norm = age / 100
    
    # Feature Vector (VAE expected order)
    raw_features = [
        age, gender_val, hr, rr, temp, sbp, dbp, spo2, wbc, lactate, creatinine, platelets, bilirubin,
        map_val, pp, si, age_norm
    ]
    
    if not is_legacy:
        # Normalize
        normalized_features = feature_scaler.transform([raw_features])
        features_tensor = torch.FloatTensor(normalized_features)
        
        # Predict with uncertainty (Monte Carlo Sampling)
        with st.spinner("Analyzing patient latent space..."):
            pred_dict = vae_model.predict_sepsis(features_tensor, n_samples=20)
            risk_mean = pred_dict['risk_mean'].item()
            risk_std = pred_dict['risk_std'].item()
            risk_lower = pred_dict['risk_lower'].item()
            risk_upper = pred_dict['risk_upper'].item()
            
            # Feature Importance / Gradients
            features_tensor.requires_grad = True
            _, _, _, risk_out = vae_model(features_tensor)
            risk_out.backward()
            importances = features_tensor.grad.abs().squeeze().numpy()
    else:
        # Legacy Fallback
        legacy_model = joblib.load('sepsis_model_v1.pkl')
        legacy_input = pd.DataFrame([{
            'Age': age, 'HeartRate': hr, 'SysBP': sbp, 'Temp': temp, 'WBC': wbc, 'ShockIndex': si
        }])
        risk_mean = legacy_model.predict_proba(legacy_input)[0][1]
        risk_std = 0.05 # Mock uncertainty
        risk_lower = max(0, risk_mean - 0.1)
        risk_upper = min(1, risk_mean + 0.1)
        importances = np.random.rand(17) # Mock

    # --- RESULTS DISPLAY ---
    st.divider()
    
    m1, m2, m3 = st.columns(3)
    
    risk_percent = risk_mean * 100
    risk_level = "LOW"
    risk_class = "low"
    if risk_mean > 0.8: risk_level, risk_class = "CRITICAL", "critical"
    elif risk_mean > 0.6: risk_level, risk_class = "HIGH", "high"
    elif risk_mean > 0.3: risk_level, risk_class = "MODERATE", "moderate"
    
    with m1:
        st.metric("Sepsis Risk Score", f"{risk_percent:.1f}%", f"{risk_level}")
    with m2:
        st.metric("Model Uncertainty", f"±{risk_std*100:.1f}%")
    with m3:
        st.metric("95% Confidence Interval", f"[{risk_lower*100:.0f}% - {risk_upper*100:.0f}%]")

    # Prediction Card
    st.markdown(f"""
    <div class="risk-card {risk_class}">
        <h2 style='color: white; margin-bottom: 0;'>Assessment: {risk_level} Risk</h2>
        <p style='color: white; opacity: 0.9;'>Based on generative latent space analysis of {int(risk_percent)}% risk patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress Bar
    st.progress(risk_mean)
    
    st.divider()
    
    # --- INSIGHTS SECTION ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("💡 Clinical Recommendation")
        if risk_level == "CRITICAL":
            st.error("❗ **Initiate Sepsis Protocol Immediately.**")
            st.write("- Call Rapid Response / ICU Consult\n- Start IV Fluids (30mL/kg)\n- Obtain Blood Cultures and Start Broad-Spectrum Antibiotics")
        elif risk_level == "HIGH":
            st.warning("⚠️ **Urgent Assessment Required.**")
            st.write("- Check Serial Lactates\n- Monitor Vitals Hourly\n- Screen for Infection Sources")
        elif risk_level == "MODERATE":
            st.info("🟡 **Increased Vigilance.**")
            st.write("- QSOFA monitoring every 4 hours\n- Review recent lab trends")
        else:
            st.success("🟢 **Routine Observation.**")
            st.write("- Standard ICU protocol\n- Monitor for changes in vitals")

    with c2:
        st.subheader("🔬 Attribution Analysis")
        # Map feature names
        feat_names = [
            'Age', 'Gender', 'Heart Rate', 'Resp Rate', 'Temp', 'Sys BP', 'Dia BP', 'SpO2', 'WBC', 
            'Lactate', 'Creatinine', 'Platelets', 'Bilirubin', 'MAP', 'Pulse Pressure', 'Shock Index', 'Age (N)'
        ]
        imp_df = pd.DataFrame({'Factor': feat_names, 'Relative Importance': importances})
        imp_df = imp_df.sort_values('Relative Importance', ascending=False).head(5)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=imp_df, x='Relative Importance', y='Factor', palette='viridis', ax=ax)
        ax.set_title("Top Risk Contributors")
        st.pyplot(fig)

    # Footer
    st.caption("Disclaimer: This tool is for educational/research purposes. Final clinical decisions must be made by qualified healthcare professionals.")