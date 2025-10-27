import streamlit as st
import pandas as pd
import joblib

# Chargement du modèle et du seuil
model_package = joblib.load("final_rf_model_with_threshold.pkl")
model = model_package["model"]
threshold = model_package["threshold"]

st.title('Fraud Detection Prediction App')
st.markdown("Please enter the transaction details and use the predict button.")
st.divider()

# Inputs
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", 'TRANSFER', 'CASH_OUT', 'DEPOSIT'])
amount = st.number_input("Amount", min_value = 0.0, value = 10000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value = 0.0, value = 10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value = 0.0, value = 10000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value = 0.0, value = 0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value = 0.0, value = 0.0)

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

     # Debug: check transformed input
    transformed = model.named_steps['add_diffs'].transform(input_data)
    st.write("Transformed Input:")
    st.write(transformed)
    
    # Vérification logique avant modèle
    logical_flag = False
    if transaction_type in ["TRANSFER", "CASH_OUT"] and newbalanceOrig > oldbalanceOrg:
        st.warning("Sender balance increased after transaction. Unusual behavior.")
        logical_flag = True
    elif transaction_type == "PAYMENT" and newbalanceOrig >= oldbalanceOrg:
        st.warning("Sender balance did not decrease after PAYMENT. Unusual behavior.")
        logical_flag = True

    # Prédiction uniquement si pas d'erreur logique critique (ou laisse passer mais signale)
    proba = model.predict_proba(input_data)[0, 1]
    prediction = int(proba >= threshold)

    st.subheader(f"Fraud Probability: {proba:.4f}")
    st.subheader(f"Prediction: **{prediction}**")

    if prediction == 1 or logical_flag:
        st.error("⚠️ This transaction is likely a fraud.")
    else:
        st.success("✅ This transaction looks safe.")