import requests
import json
import matplotlib.pyplot as plt

url = "http://localhost:5000/predict"
data_list = [
    {"IPK": 3.5, "Jumlah_Absensi": 80, "Waktu_Belajar_Jam": 10, "Rasio_Absensi": 0.9, "IPK_x_Study": 35.0},
    {"IPK": 2.5, "Jumlah_Absensi": 60, "Waktu_Belajar_Jam": 5, "Rasio_Absensi": 0.7, "IPK_x_Study": 12.5},
    {"IPK": 4.0, "Jumlah_Absensi": 95, "Waktu_Belajar_Jam": 15, "Rasio_Absensi": 0.95, "IPK_x_Study": 60.0}
]

results = []
for data in data_list:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        results.append(result)
        print("Berhasil untuk data", data, ":", result)
    else:
        print("Gagal untuk data", data, ":", response.text, "Status Code:", response.status_code)

# Simpan hasil ke file JSON
with open("prediksi_hasil.json", "w") as f:
    json.dump(results, f, indent=4)
print("Semua hasil disimpan ke prediksi_hasil.json")

# Buat dan tampilkan figure
ipk_values = [d["IPK"] for d in data_list]
prob_values = [r["probability"] for r in results if r.get("probability") is not None]

plt.figure(figsize=(8, 5))
plt.plot(ipk_values, prob_values, marker='o', linestyle='-', color='b')
plt.xlabel("IPK")
plt.ylabel("Probability")
plt.title("Hubungan IPK dengan Probability Prediksi")
plt.grid(True)
for i, txt in enumerate(prob_values):
    plt.annotate(f"{txt:.3f}", (ipk_values[i], prob_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.show()  # Tampilkan figure
print("Grafik ditampilkan di jendela baru!")