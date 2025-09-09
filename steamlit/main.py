import streamlit as st
import requests
import json

# Streamlit configuration
st.set_page_config(page_title="Bank Prediction App", page_icon="🏦", layout="centered")

# Title and description
st.title("Bank Prediction App")
st.write("Enter client details to predict the likelihood of subscribing to a term deposit.")

# Form for user input
with st.form("prediction_form"):
    st.header("Client Information")

    # Numeric inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    campaign = st.number_input("Number of Contacts (Campaign)", min_value=0, max_value=100, value=1, step=1)
    previous = st.number_input("Previous Contacts", min_value=0, value=0, step=1)
    emp_var_rate = st.number_input("Employment Variation Rate", value=1.1, step=0.1, format="%.1f")
    cons_price_idx = st.number_input("Consumer Price Index", value=93.994, step=0.1, format="%.3f")
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4, step=0.1, format="%.1f")
    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857, step=0.01, format="%.3f")
    nr_employed = st.number_input("Number of Employees", value=5191.0, step=0.1, format="%.1f")

    # Categorical inputs
    job = st.selectbox("Job", [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Has Credit in Default?", ["no", "yes"])
    housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Communication Type", ["cellular", "telephone"])

    # Submit button
    submitted = st.form_submit_button("Predict")

# Handle form submission
if submitted:
    # Prepare payload
    payload = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "campaign": campaign,
        "previous": previous,
        "emp_var_rate": emp_var_rate,
        "cons_price_idx": cons_price_idx,
        "cons_conf_idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr_employed": nr_employed
    }

    # Make API call
    with st.spinner("Making prediction..."):
        try:
            response = requests.post("http://localhost:8000/predict", json=payload)
            response.raise_for_status()
            result = response.json()

            # Display results
            st.subheader("Prediction Result")
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(f"Prediction: {result['prediction'].capitalize()}")
                st.write(f"Probability: {result['probability']:.2%}")
                st.write(f"Cached: {result['cached']}")
        except requests.RequestException as e:
            st.error(f"Failed to connect to the server: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Powered by FastAPI")