import streamlit as st
import joblib
import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import io

# Load environment variables
load_dotenv()

# Load model KNN, features, label encoder
try:
    model_knn = joblib.load('knn_model.pkl')
    features = joblib.load('features.pkl')
    le = joblib.load('label_encoder.pkl')
    st.sidebar.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")

# Konfigurasi Gemini API
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    st.sidebar.error("❌ GEMINI_API_KEY tidak ditemukan di .env")
else:
    genai.configure(api_key=gemini_api_key)
    model_gemini = genai.GenerativeModel('gemini-2.0-flash')

st.title("🍎 Deteksi Makanan & Rekomendasi Buah")

# =======================
# 1. Prediksi Manual
# =======================
st.header("🔹 Prediksi Manual")
input_data = []
for f in features:
    val = st.number_input(f, value=0.0, format="%.2f")
    input_data.append(val)

if st.button("Prediksi Manual"):
    try:
        df_input = pd.DataFrame([input_data], columns=features)
        pred = model_knn.predict(df_input)
        label_pred = le.inverse_transform(pred)[0]
        st.success(f"✅ Hasil Prediksi: {label_pred}")
    except Exception as e:
        st.error(f"❌ Error prediksi: {e}")

# =======================
# 2. Prediksi dari File CSV / Excel
# =======================
st.header("📂 Prediksi dari File")
uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Bersihkan kolom tidak perlu
        df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')

        # Cek apakah semua fitur ada
        missing_features = set(features) - set(df.columns)
        if missing_features:
            st.error(f"❌ Kolom yang hilang: {', '.join(missing_features)}")
        else:
            X = df[features].values
            preds_encoded = model_knn.predict(X)
            preds_label = le.inverse_transform(preds_encoded)
            df['Prediksi'] = preds_label
            st.success("✅ Prediksi massal berhasil!")
            st.dataframe(df)

            # Download hasil prediksi
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
            st.download_button(
                label="💾 Download Hasil (Excel)",
                data=output.getvalue(),
                file_name="hasil_prediksi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"❌ Error membaca file: {e}")

# =======================
# 3. Rekomendasi Buah dari Gemini API
# =======================
st.header("🍊 Rekomendasi Buah")
penyakit = st.text_input("Masukkan nama penyakit")

if st.button("Cari Rekomendasi Buah"):
    if not gemini_api_key:
        st.error("❌ API Key Gemini tidak tersedia.")
    elif not penyakit.strip():
        st.error("⚠ Masukkan nama penyakit terlebih dahulu.")
    else:
        try:
            prompt_text = (
                f"Saya sedang mencari rekomendasi buah untuk penyakit {penyakit}. "
                "Berikan rekomendasi satu buah yang paling cocok beserta kandungan nutrisinya. "
                "Berikan respons dalam format JSON dengan kunci 'penyakit', 'rekomendasi_buah', 'kandungan_nutrisi'. "
                "Di dalam 'kandungan_nutrisi', berikan objek JSON dengan kunci 'nama_buah', 'kalori', 'lemak', 'protein', 'karbohidrat', dan 'manfaat'. "
                "Pastikan output hanya berisi satu blok JSON saja."
            )

            response = model_gemini.generate_content(prompt_text)
            response_text = response.text.strip('`').strip('json').strip()

            try:
                info_json = json.loads(response_text)
                st.success(f"✅ Rekomendasi untuk {penyakit}")
                st.json(info_json)
            except json.JSONDecodeError:
                st.error("❌ Gagal parsing hasil dari AI. Format JSON tidak valid.")
                st.text(response_text)
        except Exception as e:
            st.error(f"❌ Error saat memproses rekomendasi: {e}")
