from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import io
import sys
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Memuat environment variables dari file .env
load_dotenv()

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Mengonfigurasi Gemini API dengan API Key dari .env
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    print("❌ Error: GEMINI_API_KEY tidak ditemukan di file .env.")
    print("❌ Pastikan Anda membuat file .env dan mengisinya dengan kunci API yang valid.")
    sys.exit()

genai.configure(api_key=gemini_api_key)

# Menggunakan model Gemini 2.0 Flash
model_gemini = genai.GenerativeModel('gemini-2.0-flash')
print("✅ Gemini API dikonfigurasi dengan model 'gemini-2.0-flash'.")

# Memuat model KNN, LabelEncoder, dan daftar fitur yang sudah dilatih
try:
    model_knn = joblib.load('knn_model.pkl')
    features = joblib.load('features.pkl')
    le = joblib.load('label_encoder.pkl')
    print("✅ Semua model dan file pendukung berhasil dimuat.")
except FileNotFoundError as e:
    print(f"❌ Error: File model, fitur, atau LabelEncoder tidak ditemukan. Error: {e}")
    print("❌ Pastikan Anda telah menjalankan skrip pelatihan model (model.py) terlebih dahulu.")
    sys.exit()

# Halaman utama aplikasi
@app.route('/')
def home():
    return render_template('index.html', features=features)

# Endpoint untuk prediksi manual dari form
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        input_data = [float(request.form[f]) for f in features]
        input_df = pd.DataFrame([input_data], columns=features)
        preds_encoded = model_knn.predict(input_df)
        preds_label = le.inverse_transform(preds_encoded)
        print(f"✅ Prediksi manual berhasil: {preds_label[0]}")
        return jsonify({"success": True, "prediction": preds_label[0]})
    except (ValueError, KeyError) as e:
        print(f"❌ Kesalahan input data dari form: {e}")
        return jsonify({"error": f"Kesalahan input data: {e}. Pastikan semua kolom terisi dengan angka."}), 400

# Endpoint untuk prediksi massal dari file CSV/Excel
@app.route('/predict_file', methods=['POST'])
def predict_file():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    filename = file.filename.lower()
    try:
        if filename.endswith('.csv'):
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            df = pd.read_csv(stream)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Format file tidak didukung. Harap upload file CSV atau Excel."}), 400
    except Exception as e:
        print(f"❌ Error membaca file: {e}")
        return jsonify({"error": f"Error membaca file: {e}"}), 400

    try:
        df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
        df_features = df[features]
    except KeyError:
        missing_features = set(features) - set(df.columns)
        print(f"❌ Error kolom file: Kolom yang hilang: {', '.join(missing_features)}")
        return jsonify({"error": f"Kolom pada file tidak sesuai. Kolom yang hilang: {', '.join(missing_features)}"}), 400

    X = df_features.values
    preds_encoded = model_knn.predict(X)
    preds_label = le.inverse_transform(preds_encoded)
    
    df['Prediksi'] = preds_label
    print("✅ Prediksi massal dari file berhasil.")
    
    return jsonify({"success": True, "results": df.to_dict('records')})

# Menggunakan Gemini API untuk rekomendasi buah DAN kandungannya
@app.route('/rekomendasi_buah', methods=['POST'])
def get_rekomendasi_buah():
    try:
        penyakit = request.form.get('penyakit')
        if not penyakit:
            return jsonify({"success": False, "error": "Input 'penyakit' tidak boleh kosong."}), 400

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
            print(f"✅ Gemini API berhasil memberikan rekomendasi untuk '{penyakit}'.")
            return jsonify({"success": True, "info": info_json})

        except json.JSONDecodeError:
            print(f"❌ Gagal mengurai respons Gemini API sebagai JSON: {response_text}")
            return jsonify({"success": False, "error": "Gagal memproses data dari AI. Format tidak valid. Pastikan prompt menghasilkan JSON yang valid."}), 500

    except Exception as e:
        print(f"❌ Error saat memproses rekomendasi buah: {e}")
        return jsonify({"success": False, "error": "Terjadi kesalahan saat mencari rekomendasi."}), 500

# Blok utama untuk menjalankan server
if __name__ == '__main__':
    print("🚀 Memulai server Flask...")
    app.run(debug=True)