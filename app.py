import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

# 1. Load Model and Metadata
@st.cache_resource
def load_assets():
    model = joblib.load('brain_hemorrhage_final_audit.pkl')
    with open('model_metadata.json', 'r') as f:
        meta = json.load(f)
    return model, meta

model, meta = load_assets()

st.set_page_config(page_title="Clinical Audit Tool", layout="wide")
st.title("🧠 Advanced Hemorrhage Risk Audit")
st.markdown("---")

# Layout: Two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.header("📋 Patient Demographics")
    age = st.slider("Patient Age", 0, 100, 50)
    gender = st.selectbox("Biological Gender", ["Female", "Male"])

    st.header("📍 Clinical Context")
    # Extract Hemorrhage Types from feature list (removing the 'Hemorrhage_Type_' prefix)
    h_types = [c.replace('Hemorrhage_Type_', '') for c in meta['features'] if 'Hemorrhage_Type_' in c]
    # Add a base case (the one that was dropped during get_dummies)
    h_type_selected = st.selectbox("Hemorrhage Type", ["Base/Other"] + h_types)

    selected_history = st.multiselect("Medical History", meta['history'])

with col2:
    st.header("🌡️ Acute Presentation")
    selected_symptoms = st.multiselect("Current Symptoms", meta['symptoms'])

    st.info("""
    **Statistical Note:**
    This tool utilizes an ensemble model (XGBoost + Random Forest) calibrated via Isotonic Regression.
    Gender and Hemorrhage Type are weighted based on population-level Odds Ratios.
    """)

if st.button("RUN CLINICAL AUDIT", use_container_width=True):
    # 3. Create the input row with the EXACT features the model expects
    input_df = pd.DataFrame(0, index=[0], columns=meta['features'])

    # Map Demographic Inputs
    input_df['Age'] = age
    if f"Gender_{gender}" in input_df.columns:
        input_df[f"Gender_{gender}"] = 1

    # Map Hemorrhage Type
    if f"Hemorrhage_Type_{h_type_selected}" in input_df.columns:
        input_df[f"Hemorrhage_Type_{h_type_selected}"] = 1

    # Map Symptoms & History
    for s in selected_symptoms:
        f_name = s.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        if f_name in input_df.columns: input_df[f_name] = 1

    for h in selected_history:
        f_name = h.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        if f_name in input_df.columns: input_df[f_name] = 1

    # Interaction Term (Age x Hypertension)
    is_hypertensive = 1 if "Hypertension" in selected_history else 0
    if 'Age_x_Hypertension' in input_df.columns:
        input_df['Age_x_Hypertension'] = age * is_hypertensive

    # 5. Prediction
    prob = model.predict_proba(input_df)[0, 1]

    # 6. Display Result with Clinical Interpretation
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric("Mortality Risk", f"{prob:.2%}")
        if prob >= meta['clinical_threshold']:
            st.error("🚨 HIGH RISK")
        else:
            st.success("✅ STABLE")

    with res_col2:
        st.write("### Clinical Justification")
        st.write(f"The threshold for emergency intervention is set at **{meta['clinical_threshold']:.2%}**.")

        # Demonstrating influence (Rationale)
        influence_factors = []
        if age > 65: influence_factors.append("Advanced Age (increasing arterial fragility)")
        if h_type_selected != "Base/Other": influence_factors.append(f"Specific Hemorrhage Type ({h_type_selected})")
        if is_hypertensive: influence_factors.append("Hypertension Synergy (captured via interaction term)")

        if influence_factors:
            st.write("**Key factors influencing this score:**")
            for factor in influence_factors:
                st.write(f"- {factor}")
