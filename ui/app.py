
import streamlit as st
import joblib
import pandas as pd
import re
from urllib.parse import urlparse

def extract_features(entry, entry_type):
    f = {
        'length': len(entry),
        'has_https': int(entry.startswith('https')) if entry_type == 'url' else 0,
        'num_dots': entry.count('.'),
        'has_at': int('@' in entry),
        'has_hyphen': int('-' in entry),
        'has_ip': int(bool(re.search(r'(\d+\.\d+\.\d+\.\d+)', entry))),
        'suspicious_words': sum(w in entry.lower() for w in ['login', 'verify', 'secure', 'update', 'confirm']),
        'path_length': len(urlparse(entry).path) if entry_type == 'url' else 0
    }
    return pd.DataFrame([f])

model = joblib.load("models/rf_model.pkl")

st.set_page_config(page_title="Phishing Detector", layout="centered")
st.title("🛡️ Phishing URL/Email Detection")

entry = st.text_input("Enter a URL or Email ID:")
entry_type = st.radio("Select Type", ["url", "email"])

if st.button("Analyze"):
    if entry:
        features = extract_features(entry, entry_type)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]
        st.subheader("🔍 Prediction")
        if pred == 1:
            st.error(f"⚠️ This is likely a *Phishing* attempt! ({prob*100:.2f}% confidence)")
        else:
            st.success(f"✅ This appears to be *Legitimate*. ({prob*100:.2f}% confidence)")

        st.subheader("📊 Feature Breakdown")
        st.bar_chart(features.T.rename(columns={0: "Value"}))
    else:
        st.warning("Please enter a value.")
