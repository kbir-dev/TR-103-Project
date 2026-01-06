import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import shap
from scipy.sparse import issparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Healthcare Cost Prediction & Explanation",
    page_icon="💊",
    layout="wide"
)

# Custom CSS for better text display
st.markdown("""
<style>
    .stTextArea textarea {
        font-family: 'Arial', sans-serif !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        color: #1a1a1a !important;
        background-color: #f0f2f6 !important;
        border: 2px solid #4a4a4a !important;
        padding: 15px !important;
    }
    
    /* Force text color in disabled text areas */
    .stTextArea textarea[disabled] {
        color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODELS AND PREPROCESSOR
# -------------------------------------------------
@st.cache_resource
def load_models():
    """Load ML models and preprocessor"""
    try:
        # Check if model files exist
        if not os.path.exists("models/reg_preprocessor.pkl"):
            st.error("❌ Preprocessor file not found: models/reg_preprocessor.pkl")
            return None, None, None
        
        # Load the preprocessor
        preprocessor = joblib.load("models/reg_preprocessor.pkl")
        
        # Load the models - try XGBoost first, fallback to Random Forest
        model = None
        model_name = None
        
        if os.path.exists("models/xgboost_reg.pkl"):
            try:
                model = joblib.load("models/xgboost_reg.pkl")
                model_name = "XGBoost"
            except Exception as e:
                st.warning(f"Could not load XGBoost model: {str(e)}. Trying Random Forest...")
        
        if model is None and os.path.exists("models/random_forest_reg.pkl"):
            try:
                model = joblib.load("models/random_forest_reg.pkl")
                model_name = "Random Forest"
            except Exception as e:
                st.error(f"Could not load Random Forest model: {str(e)}")
        
        if model is None:
            st.error("❌ No model files found. Please ensure models are in the models/ directory.")
            return None, None, None
            
        return model, preprocessor, model_name
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None

# Load models at startup
model, preprocessor, model_name = load_models()

# -------------------------------------------------
# GLOBAL VARIABLES
# -------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Define expected columns for CSV upload (all features required by the preprocessor)
EXPECTED_COLUMNS = [
    # Basic demographics
    "age", "sex", "region", "urban_rural", "income", "education", 
    "marital_status", "employment_status", "household_size", "dependents",
    # Health metrics
    "bmi", "smoker", "alcohol_freq", "visits_last_year",
    "hospitalizations_last_3yrs", "days_hospitalized_last_3yrs", "medication_count",
    "systolic_bp", "diastolic_bp", "ldl", "hba1c",
    # Insurance info
    "plan_type", "network_tier", "deductible", "copay", 
    "policy_term_years", "policy_changes_last_2yrs", "provider_quality",
    # Risk factors
    "risk_score", "chronic_count",
    # Chronic conditions (binary: 0 or 1)
    "hypertension", "diabetes", "asthma", "copd", "cardiovascular_disease",
    "cancer_history", "kidney_disease", "liver_disease", "arthritis", "mental_health",
    # Procedure counts
    "proc_imaging_count", "proc_surgery_count", "proc_physio_count",
    "proc_consult_count", "proc_lab_count",
    # Risk flags (binary: 0 or 1)
    "is_high_risk", "had_major_procedure"
]

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def clean_feature_name(feature_name):
    """Clean feature names for better display (remove prefixes like 'num__', 'cat__')"""
    # Remove common prefixes
    if feature_name.startswith('num__'):
        return feature_name.replace('num__', '')
    elif feature_name.startswith('cat__'):
        # For one-hot encoded features, format them nicely
        # e.g., "cat__smoker_Current" -> "Smoker: Current"
        name = feature_name.replace('cat__', '')
        if '_' in name:
            parts = name.split('_', 1)
            return f"{parts[0].title()}: {parts[1]}"
        return name.title()
    return feature_name

# Create a sample CSV template for download
def get_sample_csv():
    sample_data = pd.DataFrame([
        {
            # Basic demographics
            "age": 35,
            "sex": "Male",
            "region": "North",
            "urban_rural": "Urban",
            "income": 60000,
            "education": "Bachelors",
            "marital_status": "Married",
            "employment_status": "Employed",
            "household_size": 3,
            "dependents": 1,
            # Health metrics
            "bmi": 26.0,
            "smoker": "Never",
            "alcohol_freq": "Occasional",
            "visits_last_year": 2,
            "hospitalizations_last_3yrs": 0,
            "days_hospitalized_last_3yrs": 0,
            "medication_count": 2,
            "systolic_bp": 120.0,
            "diastolic_bp": 80.0,
            "ldl": 100.0,
            "hba1c": 5.5,
            # Insurance info
            "plan_type": "PPO",
            "network_tier": "Silver",
            "deductible": 1000,
            "copay": 20,
            "policy_term_years": 5,
            "policy_changes_last_2yrs": 0,
            "provider_quality": 3.5,
            # Risk factors
            "risk_score": 2.5,
            "chronic_count": 1,
            # Chronic conditions (0 or 1)
            "hypertension": 0,
            "diabetes": 0,
            "asthma": 0,
            "copd": 0,
            "cardiovascular_disease": 0,
            "cancer_history": 0,
            "kidney_disease": 0,
            "liver_disease": 0,
            "arthritis": 0,
            "mental_health": 0,
            # Procedure counts
            "proc_imaging_count": 1,
            "proc_surgery_count": 0,
            "proc_physio_count": 0,
            "proc_consult_count": 2,
            "proc_lab_count": 3,
            # Risk flags (0 or 1)
            "is_high_risk": 0,
            "had_major_procedure": 0
        }
    ])
    return sample_data

# Prediction function using the loaded ML model
def predict_cost(data):
    """
    Predict healthcare cost based on input features using the ML model.
    Returns prediction and feature importances.
    """
    if model is None or preprocessor is None:
        # Fallback to mock prediction if models failed to load
        return mock_predict_cost(data)
    
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure all required columns are present
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in input_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Reorder columns to match preprocessor expectations
        input_df = input_df[EXPECTED_COLUMNS]
        
        # Preprocess the data
        X = preprocessor.transform(input_df)
        
        # Convert sparse matrix to dense if needed
        if issparse(X):
            X = X.toarray()
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Calculate SHAP values for feature importance
        # Use TreeExplainer for tree-based models (XGBoost, Random Forest)
        if model_name in ["XGBoost", "Random Forest"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            # Fallback to general Explainer for other models
            explainer = shap.Explainer(model)
            shap_values = explainer.shap_values(X)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For some models
        if len(shap_values.shape) > 2:
            shap_values = shap_values[0]  # For multi-output models
        
        # Get feature names from preprocessor
        try:
            num_cols = preprocessor.named_transformers_["num"].get_feature_names_out()
            cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out()
            feature_names = np.concatenate([num_cols, cat_cols])
        except:
            # Fallback if get_feature_names_out is not available
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            else:
                num_features = X.shape[1]
                feature_names = [f'feature_{i}' for i in range(num_features)]
        
        # Create dictionary of feature importances with cleaned names
        feature_importance = {}
        for i, name in enumerate(feature_names):
            if i < shap_values.shape[1]:
                clean_name = clean_feature_name(name)
                feature_importance[clean_name] = shap_values[0, i]
        
        # Sort and get top features
        sorted_features = {k: v for k, v in sorted(
            feature_importance.items(), 
            key=lambda item: abs(item[1]), 
            reverse=True
        )[:10]}  # Get top 10 features
        
        return prediction, sorted_features
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        # Fallback to mock prediction
        return mock_predict_cost(data)

# Mock prediction function as fallback
def mock_predict_cost(data):
    """
    Fallback mock prediction function.
    """
    base_cost = 5000
    
    # Age factor (older = higher cost)
    age_factor = data['age'] / 30
    
    # BMI factor (higher BMI = higher cost)
    bmi_factor = 1.0
    if data['bmi'] > 30:
        bmi_factor = 1.5
    elif data['bmi'] > 25:
        bmi_factor = 1.2
    
    # Smoker factor
    smoker_factor = 1.0
    if data['smoker'] == 'Current':
        smoker_factor = 2.0
    elif data['smoker'] == 'Former':
        smoker_factor = 1.3
    
    # Visits factor
    visits_factor = 1.0 + (data['visits_last_year'] * 0.05)
    
    # Network tier discount
    tier_discount = 1.0
    if data['network_tier'] == 'Platinum':
        tier_discount = 0.8
    elif data['network_tier'] == 'Gold':
        tier_discount = 0.85
    elif data['network_tier'] == 'Silver':
        tier_discount = 0.9
    
    # Calculate final cost
    cost = base_cost * age_factor * bmi_factor * smoker_factor * visits_factor * tier_discount
    
    # Generate feature importance (mock SHAP values)
    features = {
        'age': (age_factor - 1) * base_cost * bmi_factor * smoker_factor * visits_factor * tier_discount,
        'bmi': (bmi_factor - 1) * base_cost * age_factor * smoker_factor * visits_factor * tier_discount,
        'smoker_status': (smoker_factor - 1) * base_cost * age_factor * bmi_factor * visits_factor * tier_discount,
        'visits_last_year': (visits_factor - 1) * base_cost * age_factor * bmi_factor * smoker_factor * tier_discount,
        'network_tier': (tier_discount - 1) * base_cost * age_factor * bmi_factor * smoker_factor * visits_factor
    }
    
    return cost, features

# -------------------------------------------------
# LLM via OPENROUTER (DEEPSEEK)
# -------------------------------------------------
def call_deepseek_openrouter(user_data, prediction, top_features):
    """Call DeepSeek LLM via OpenRouter to generate explanation."""
    if not OPENROUTER_API_KEY:
        return "⚠️ LLM disabled: OPENROUTER_API_KEY not set. Please add it to your .env file."

    url = "https://openrouter.ai/api/v1/chat/completions"

    # Format feature importance for LLM
    feat_text = "\n".join(
        [f"- {name}: ${value:.2f}" for name, value in top_features.items() if abs(value) > 0]
    )

    system_prompt = (
        "You are a professional healthcare insurance analyst. "
        "Explain model-predicted annual medical costs clearly and simply. "
        "Do NOT diagnose or give medical treatment. Focus on cost factors only."
    )

    user_prompt = f"""
User Information:
{json.dumps(user_data, indent=2)}

Predicted Annual Medical Cost: ${prediction:.2f}

Top Cost Factors:
{feat_text}

Explain:
1. Why the predicted cost is at this level.
2. Which factors increased the cost.
3. Which factors helped reduce the cost.
4. Give 3-4 simple lifestyle or preventive tips (non-medical).
5. Keep the explanation friendly (~200 words).
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app.com",
        "X-Title": "Health Cost XAI App"
    }

    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 600,
        "temperature": 0.4
    }

    try:
        res = requests.post(url, headers=headers, json=data, timeout=60)
        res.raise_for_status()
        llm_response = res.json()["choices"][0]["message"]["content"]
        return llm_response
    except requests.exceptions.RequestException as e:
        return f"❌ Error calling LLM: {str(e)}\n\nPlease check your API key and internet connection."
    except (KeyError, IndexError) as e:
        return f"❌ Error parsing LLM response: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

# -------------------------------------------------
# MAIN UI WITH TABS
# -------------------------------------------------
st.title("💊 Healthcare Cost Prediction + XAI + LLM Insight")
if model_name:
    st.info(f"🤖 Using **{model_name}** model for predictions")
st.write("This app will:")
st.markdown("""
1. Predict your **annual medical cost**  
2. Generate **feature importance visualization** using SHAP  
3. Ask **DeepSeek (LLM)** to summarize your risks in simple words  
""")

st.markdown("---")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📝 Manual Input", "📤 Upload CSV"])

# Initialize session state for input data
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'should_predict' not in st.session_state:
    st.session_state.should_predict = False

# Tab 1: Manual Input Form
with tab1:
    st.info("Please fill in all the fields below. Use the expandable sections to organize your information.")
    
    # Basic Demographics Section
    with st.expander("👤 Basic Demographics", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("📅 Age", 18, 100, 35, key="manual_age")
            sex = st.selectbox("⚧ Gender", ["Male", "Female"], key="manual_sex")
            region = st.selectbox("🌍 Region", ["North", "South", "East", "West", "Central"], key="manual_region")
        with col2:
            urban_rural = st.selectbox("🌉 Living Area", ["Urban", "Rural", "Suburban"], key="manual_urban_rural")
            income = st.number_input("💰 Annual Income ($)", min_value=0, max_value=500000, value=60000, step=1000, key="manual_income")
            education = st.selectbox("🎓 Education", ["No HS", "HS", "Some College", "Bachelors", "Masters", "Doctorate"], key="manual_education")
        with col3:
            marital_status = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced", "Widowed"], key="manual_marital_status")
            employment_status = st.selectbox("💼 Employment", ["Employed", "Unemployed", "Self-employed", "Retired"], key="manual_employment_status")
            household_size = st.number_input("👨‍👩‍👧‍👦 Household Size", 1, 10, 3, key="manual_household_size")
            dependents = st.number_input("👶 Dependents", 0, 10, 1, key="manual_dependents")
    
    # Health Metrics Section
    with st.expander("🏥 Health Metrics", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            bmi = st.number_input("⚖️ BMI", 10.0, 60.0, 26.0, step=0.1, key="manual_bmi")
            smoker = st.selectbox("🚬 Smoking Status", ["Never", "Former", "Current"], key="manual_smoker")
            alcohol_freq = st.selectbox("🍺 Alcohol Frequency", ["None", "Occasional", "Weekly", "Daily"], key="manual_alcohol_freq")
            visits_last_year = st.number_input("🏥 Doctor Visits (Last Year)", 0, 50, 2, key="manual_visits_last_year")
        with col2:
            hospitalizations = st.number_input("🏥 Hospitalizations (Last 3 Years)", 0, 20, 0, key="manual_hospitalizations")
            days_hospitalized = st.number_input("📅 Days Hospitalized (Last 3 Years)", 0, 365, 0, key="manual_days_hospitalized")
            medication_count = st.number_input("💊 Medication Count", 0, 20, 2, key="manual_medication_count")
            risk_score = st.number_input("⚠️ Risk Score", 0.0, 10.0, 2.5, step=0.1, key="manual_risk_score")
        with col3:
            systolic_bp = st.number_input("🩺 Systolic BP", 80.0, 200.0, 120.0, step=1.0, key="manual_systolic_bp")
            diastolic_bp = st.number_input("🩺 Diastolic BP", 40.0, 120.0, 80.0, step=1.0, key="manual_diastolic_bp")
            ldl = st.number_input("🩺 LDL Cholesterol", 0.0, 300.0, 100.0, step=1.0, key="manual_ldl")
            hba1c = st.number_input("🩺 HbA1c", 3.0, 15.0, 5.5, step=0.1, key="manual_hba1c")
    
    # Chronic Conditions Section
    with st.expander("🩺 Chronic Conditions (Check if you have)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            hypertension = st.checkbox("Hypertension", key="manual_hypertension")
            diabetes = st.checkbox("Diabetes", key="manual_diabetes")
            asthma = st.checkbox("Asthma", key="manual_asthma")
            copd = st.checkbox("COPD", key="manual_copd")
        with col2:
            cardiovascular_disease = st.checkbox("Cardiovascular Disease", key="manual_cardiovascular")
            cancer_history = st.checkbox("Cancer History", key="manual_cancer")
            kidney_disease = st.checkbox("Kidney Disease", key="manual_kidney")
            liver_disease = st.checkbox("Liver Disease", key="manual_liver")
        with col3:
            arthritis = st.checkbox("Arthritis", key="manual_arthritis")
            mental_health = st.checkbox("Mental Health Condition", key="manual_mental_health")
            chronic_count = st.number_input("Total Chronic Conditions Count", 0, 10, 0, key="manual_chronic_count")
    
    # Procedures Section
    with st.expander("🔬 Medical Procedures (Last Year)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            proc_imaging_count = st.number_input("📷 Imaging Procedures", 0, 50, 1, key="manual_proc_imaging")
            proc_surgery_count = st.number_input("🔪 Surgeries", 0, 20, 0, key="manual_proc_surgery")
            proc_physio_count = st.number_input("💪 Physical Therapy", 0, 50, 0, key="manual_proc_physio")
        with col2:
            proc_consult_count = st.number_input("👨‍⚕️ Consultations", 0, 50, 2, key="manual_proc_consult")
            proc_lab_count = st.number_input("🧪 Lab Tests", 0, 50, 3, key="manual_proc_lab")
    
    # Insurance & Policy Section
    with st.expander("📝 Insurance & Policy Info", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            plan_type = st.selectbox("📋 Plan Type", ["HMO", "PPO", "EPO", "POS"], key="manual_plan_type")
            network_tier = st.selectbox("⭐ Network Tier", ["Bronze", "Silver", "Gold", "Platinum"], key="manual_network_tier")
        with col2:
            deductible = st.number_input("💳 Deductible ($)", 0, 20000, 1000, step=500, key="manual_deductible")
            copay = st.number_input("💵 Copay per Visit ($)", 0, 2000, 20, step=10, key="manual_copay")
            policy_term_years = st.number_input("📆 Policy Term (Years)", 1, 10, 5, key="manual_policy_term_years")
        with col3:
            policy_changes = st.number_input("🔁 Policy Changes (Last 2 Years)", 0, 10, 0, key="manual_policy_changes")
            provider_quality = st.slider("🏥 Provider Quality (1-5)", 1.0, 5.0, 3.5, 0.1, key="manual_provider_quality")
    
    # Risk Flags Section
    with st.expander("⚠️ Risk Flags", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            is_high_risk = st.checkbox("High Risk Patient", key="manual_is_high_risk")
        with col2:
            had_major_procedure = st.checkbox("Had Major Procedure", key="manual_had_major_procedure")

    # Build DF for manual input
    if st.button("🔮 Predict & Explain", key="manual_predict_button", type="primary"):
        st.session_state.input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "region": region,
            "urban_rural": urban_rural,
            "income": income,
            "education": education,
            "marital_status": marital_status,
            "employment_status": employment_status,
            "household_size": household_size,
            "dependents": dependents,
            "bmi": bmi,
            "smoker": smoker,
            "alcohol_freq": alcohol_freq,
            "visits_last_year": visits_last_year,
            "hospitalizations_last_3yrs": hospitalizations,
            "days_hospitalized_last_3yrs": days_hospitalized,
            "medication_count": medication_count,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "ldl": ldl,
            "hba1c": hba1c,
            "plan_type": plan_type,
            "network_tier": network_tier,
            "deductible": deductible,
            "copay": copay,
            "policy_term_years": policy_term_years,
            "policy_changes_last_2yrs": policy_changes,
            "provider_quality": provider_quality,
            "risk_score": risk_score,
            "chronic_count": chronic_count,
            "hypertension": 1 if hypertension else 0,
            "diabetes": 1 if diabetes else 0,
            "asthma": 1 if asthma else 0,
            "copd": 1 if copd else 0,
            "cardiovascular_disease": 1 if cardiovascular_disease else 0,
            "cancer_history": 1 if cancer_history else 0,
            "kidney_disease": 1 if kidney_disease else 0,
            "liver_disease": 1 if liver_disease else 0,
            "arthritis": 1 if arthritis else 0,
            "mental_health": 1 if mental_health else 0,
            "proc_imaging_count": proc_imaging_count,
            "proc_surgery_count": proc_surgery_count,
            "proc_physio_count": proc_physio_count,
            "proc_consult_count": proc_consult_count,
            "proc_lab_count": proc_lab_count,
            "is_high_risk": 1 if is_high_risk else 0,
            "had_major_procedure": 1 if had_major_procedure else 0
        }])
        st.session_state.should_predict = True
        st.rerun()

# Tab 2: CSV Upload
with tab2:
    st.markdown("### Upload a CSV file with patient data")
    st.info("The CSV should contain one row with all the required fields. Download the template below for reference.")
    
    # Create a sample CSV for download
    sample_csv = get_sample_csv()
    csv_download = sample_csv.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_download,
        file_name="healthcare_data_template.csv",
        mime="text/csv"
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            csv_data = pd.read_csv(uploaded_file)
            
            # Check if the CSV has the expected columns
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in csv_data.columns]
            
            if missing_cols:
                st.error(f"Missing columns in CSV: {', '.join(missing_cols)}")
                st.session_state.input_data = None
            elif len(csv_data) == 0:
                st.error("The uploaded CSV file is empty.")
                st.session_state.input_data = None
            else:
                if len(csv_data) > 1:
                    st.warning("Multiple rows detected. Only the first row will be used.")
                
                st.session_state.input_data = csv_data.iloc[[0]].copy()
                
                # Display the uploaded data
                st.subheader("Uploaded Data Preview:")
                st.dataframe(st.session_state.input_data.transpose(), height=400)
                
                # Predict button for CSV upload
                if st.button("🔮 Predict & Explain", key="csv_predict_button"):
                    st.session_state.should_predict = True
                    st.rerun()
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.session_state.input_data = None

st.markdown("---")

# -------------------------------------------------
# RESULTS SECTION
# -------------------------------------------------
# Only show results if input_data is available and predict button was clicked
if st.session_state.input_data is not None and st.session_state.should_predict:
    with st.spinner("Processing your data…"):
        try:
            # Get user data as dictionary
            user_data = st.session_state.input_data.iloc[0].to_dict()

            # Predict cost using our simplified model
            prediction, feature_importances = predict_cost(user_data)

            # Create a results container
            st.subheader("💰 Predicted Annual Medical Cost")
            st.metric("Estimated Cost", f"${prediction:,.2f}")

            # Sort features by importance
            sorted_features = dict(sorted(feature_importances.items(), 
                                         key=lambda x: abs(x[1]), 
                                         reverse=True))

            # Display feature importance
            st.subheader("🧠 Top Feature Contributions")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get names and values for plotting
            names = list(sorted_features.keys())[::-1]
            vals = list(sorted_features.values())[::-1]
            
            # Color bars based on positive/negative impact
            colors = ['green' if v < 0 else 'red' for v in vals]

            bars = ax.barh(names, vals, color=colors, alpha=0.7)
            ax.set_title("Top Factors Impacting Your Prediction", fontsize=14, fontweight='bold')
            ax.set_xlabel("Effect on Annual Cost ($)", fontsize=12)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            # LLM Explanation
            st.subheader("🤖 Personalized Explanation (DeepSeek LLM)")
            with st.spinner("Generating natural-language explanation…"):
                explanation = call_deepseek_openrouter(
                    user_data=user_data,
                    prediction=prediction,
                    top_features=sorted_features
                )

            # Display the explanation with asterisks removed and text wrapping
            st.subheader("📝 AI Analysis")
            # Remove asterisks from the explanation
            clean_explanation = explanation.replace('*', '')
            
            # Use custom CSS to create a text box with word wrapping
            st.markdown("""
            <style>
            .wrapped-text {
                background-color: #f0f2f6;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 15px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                color: #1e1e1e;
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
                line-height: 1.5;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display the text with wrapping
            # Escape HTML characters to prevent rendering issues
            import html
            escaped_explanation = html.escape(clean_explanation)
            
            # Replace newlines with <br> tags to preserve line breaks
            formatted_explanation = escaped_explanation.replace('\n', '<br>')
            
            st.markdown(f'<div class="wrapped-text">{formatted_explanation}</div>', unsafe_allow_html=True)
            
            # Reset the prediction flag
            st.session_state.should_predict = False
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.session_state.should_predict = False