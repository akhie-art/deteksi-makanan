import streamlit as st
import joblib
import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Load model
model_knn = joblib.load('knn_model.pkl')
features = joblib.load('features.pkl')
le = joblib.load('label_encoder.pkl')

# Gemini config
gemini_api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

st.title("Deteksi Makanan & Rekomendasi Buah")

# Input manual
st.subheader("Prediksi Manual")
input_data = []
for f in features:
    val = st.number_input(f, value=0.0)
    input_data.append(val)

if st.button("Prediksi"):
    df = pd.DataFrame([input_data], columns=features)
    pred = model_knn.predict(df)
    st.success(f"Hasil Prediksi: {le.inverse_transform(pred)[0]}")

# Rekomendasi buah
st.subheader("Rekomendasi Buah")
penyakit = st.text_input("Masukkan nama penyakit")
if st.button("Cari Rekomendasi Buah"):
    prompt_text = (
        f"Saya sedang mencari rekomendasi buah untuk penyakit {penyakit}. "
        "Berikan rekomendasi satu buah yang paling cocok beserta kandungan nutrisinya. "
        "Berikan respons dalam format JSON..."
    )
    response = model_gemini.generate_content(prompt_text)
    try:
        info = json.loads(response.text.strip('`').strip('json').strip())
        st.json(info)
    except:
        st.error("Format respons AI tidak valid")
