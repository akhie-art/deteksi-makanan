import streamlit as st
import joblib
import pandas as pd
import re
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

# ======================
# Load environment
# ======================
load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY tidak ditemukan di file .env")
    st.stop()

genai.configure(api_key=gemini_api_key)
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

# ======================
# Load model & features
# ======================
try:
    model_knn = joblib.load('knn_model.pkl')
    features = joblib.load('features.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError as e:
    st.error(f"❌ File model tidak ditemukan: {e}")
    st.stop()

# ======================
# Fungsi prediksi
# ======================
def predict_manual(input_data):
    df = pd.DataFrame([input_data], columns=features)
    pred_encoded = model_knn.predict(df)
    return le.inverse_transform(pred_encoded)[0]

def predict_file(df):
    df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    df_features = df[features]
    preds_encoded = model_knn.predict(df_features)
    df['Prediksi'] = le.inverse_transform(preds_encoded)
    return df

def rekomendasi_buah(penyakit):
    prompt_text = (
        f"Saya sedang mencari rekomendasi buah untuk penyakit {penyakit}. "
        "Berikan hanya satu buah terbaik dan kandungan nutrisinya. "
        "Balas HANYA dalam format JSON valid dengan kunci: "
        "'penyakit', 'rekomendasi_buah', 'kandungan_nutrisi' "
        "(yang berisi 'nama_buah', 'kalori', 'lemak', 'protein', 'karbohidrat', 'manfaat')."
    )
    response = model_gemini.generate_content(prompt_text)
    raw_text = response.text.strip()

    # Ambil JSON dari teks
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not json_match:
        return None, "Tidak menemukan JSON di respons AI."
    
    try:
        clean_json = json.loads(json_match.group(0))
        return clean_json, None
    except json.JSONDecodeError as e:
        return None, f"Format JSON tidak valid: {e}"

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Prediksi Makanan & Rekomendasi Buah", layout="wide")
st.title("🍎 Prediksi Makanan & Rekomendasi Buah")

menu = st.sidebar.radio("Pilih Menu", ["Prediksi Manual", "Prediksi dari File", "Rekomendasi Buah"])

if menu == "Prediksi Manual":
    st.subheader("🔍 Prediksi Manual")
    input_data = []
    for f in features:
        val = st.number_input(f, value=0.0)
        input_data.append(val)
    
    if st.button("Prediksi"):
        try:
            hasil = predict_manual(input_data)
            st.success(f"Hasil Prediksi: **{hasil}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

elif menu == "Prediksi dari File":
    st.subheader("📂 Prediksi dari File CSV/XLSX")
    file = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if file:
        try:
            if file.name.lower().endswith('.csv'):
                df = pd.read_csv(file)
            else:
                import openpyxl
                df = pd.read_excel(file)
            
            hasil_df = predict_file(df)
            st.dataframe(hasil_df)
            
            # Download hasil
            csv = hasil_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil CSV", csv, "hasil_prediksi.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

elif menu == "Rekomendasi Buah":
    st.subheader("🍌 Rekomendasi Buah")
    penyakit = st.text_input("Masukkan nama penyakit")
    if st.button("Cari Rekomendasi"):
        if not penyakit.strip():
            st.warning("Masukkan nama penyakit terlebih dahulu.")
        else:
            hasil, err = rekomendasi_buah(penyakit)
            if err:
                st.error(err)
            else:
                st.json(hasil)
