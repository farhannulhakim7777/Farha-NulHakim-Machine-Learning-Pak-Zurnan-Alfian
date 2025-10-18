from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sys
import os

app = Flask(__name__)

# Log untuk debug
print("Memulai aplikasi Flask...")
print(f"Direktori kerja: {os.getcwd()}")
print(f"File model.pkl ada? {os.path.exists('model.pkl')}")

try:
    # Cek dan load model
    if not os.path.exists("model.pkl"):
        raise FileNotFoundError("File model.pkl tidak ditemukan di folder ini. Pastikan file ada di direktori yang sama!")
    MODEL = joblib.load("model.pkl")
    print("Model dimuat berhasil.")
    
    # Auto-detect expected columns dari model (kalau sklearn atau sejenis)
    if hasattr(MODEL, 'feature_names_in_'):
        EXPECTED_COLUMNS = list(MODEL.feature_names_in_)
        print(f"Expected columns otomatis dari model: {EXPECTED_COLUMNS}")
    else:
        # ISI MANUAL DI SINI SESUAI PROYEK ANDA! 
        # Contoh: EXPECTED_COLUMNS = ['umur', 'gaji', 'pendidikan']
        # Ambil dari training script (X_train.columns.tolist() atau sejenis)
        EXPECTED_COLUMNS = []  # <-- GANTI INI DENGAN LIST KOLOM FEATURES YANG BENAR!
        print(f"Expected columns manual (isi dulu!): {EXPECTED_COLUMNS}")
        if not EXPECTED_COLUMNS:
            raise ValueError("EXPECTED_COLUMNS kosong! Isi manual sesuai model training di kode di atas.")
        
except Exception as e:
    print(f"Error memuat model: {str(e)}", file=sys.stderr)
    print("Pastikan model.pkl ada, library kompatibel, dan isi EXPECTED_COLUMNS.", file=sys.stderr)
    sys.exit(1)

@app.route("/predict", methods=["POST"])
def predict():
    print("Request diterima:", request.get_json())  # Debug di terminal
    try:
        data = request.get_json(force=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Data JSON tidak valid atau kosong. Kirim dict seperti {'col1': value1, 'col2': value2}"}), 400

        # Cek missing columns
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in data]
        if missing_cols:
            return jsonify({"error": f"Kolom yang hilang: {missing_cols}. Harus kirim semua kolom: {EXPECTED_COLUMNS}"}), 400

        # Buat DataFrame dari data, reorder ke expected columns (biar match urutan model)
        X = pd.DataFrame([data])[EXPECTED_COLUMNS]
        
        # Convert semua ke numeric (hindari error tipe data, misal string jadi NaN)
        X = X.apply(pd.to_numeric, errors='coerce')
        if X.isnull().any().any():
            return jsonify({"error": "Semua nilai data harus berupa angka (numeric)! Cek input JSON Anda."}), 400

        # Prediksi
        yhat = MODEL.predict(X)[0]
        proba = None
        if hasattr(MODEL, "predict_proba"):
            # Asumsi binary classification, ambil prob class 1 (indeks 1)
            proba = float(MODEL.predict_proba(X)[:, 1][0])
            print(f"Prediksi internal: {yhat}, Probabilitas class 1: {proba}")  # Debug
        
        return jsonify({
            "prediction": int(yhat), 
            "probability": proba,
            "expected_columns": EXPECTED_COLUMNS  # Info tambahan buat client
        })
    except Exception as e:
        print(f"Error di fungsi predict: {str(e)}")  # Debug di terminal
        return jsonify({"error": f"Error dalam prediksi: {str(e)}. Cek data input, kolom, dan tipe data."}), 500

# INI YANG PENTING: Start Flask server biar app bisa dijalankan!
if __name__ == "__main__":
    print("\nFlask server akan start di http://127.0.0.1:5000")
    print("Tekan Ctrl+C untuk stop server.")
    print("Test endpoint dengan POST ke /predict menggunakan curl atau Postman.\n")
    app.run(debug=True, host='127.0.0.1', port=5000)