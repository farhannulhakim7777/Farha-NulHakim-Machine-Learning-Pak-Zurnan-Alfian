# =================================================================
# LANGKAH 1: PERSIAPAN DATA (REVISI ERROR STRATIFIKASI)
# =================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Memuat data
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Scaling Fitur
sc = StandardScaler()
Xs = sc.fit_transform(X)

# Pembagian 1: Train (70%) dan Temp (30%). Menggunakan stratify=y.
X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42)

# Pembagian 2: Val (15%) dan Test (15%) dari Temp.
# REVISI: Menghapus stratify=y_temp untuk menghindari ValueError jika kelas minoritas terlalu sedikit.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42) # stratify=y_temp DIHAPUS

print("--- Ukuran Data Siap ---")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print("-" * 30)

# =================================================================
# LANGKAH 2 & 3: MEMBANGUN, KOMPILASI, DAN MELATIH MODEL BASELINE
# =================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Membangun Model
model_baseline = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Kompilasi Model
model_baseline.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", "AUC"])
print("--- Model Baseline Summary ---")
model_baseline.summary()
print("-" * 30)

# Callback Early Stopping
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Melatih Model
print("--- Pelatihan Model Baseline Dimulai ---")
history = model_baseline.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[es], verbose=1 # verbose=1 menampilkan progress
)
print("Pelatihan Selesai. Epoch terbaik ditemukan.")
print("-" * 30)

# =================================================================
# LANGKAH 4: EVALUASI MODEL BASELINE PADA DATA TEST
# =================================================================
from sklearn.metrics import classification_report, confusion_matrix

print("--- Evaluasi Model Baseline ---")
loss, acc, auc = model_baseline.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test AUC: {auc:.4f}")

# Prediksi dan Konversi ke Kelas
y_proba = model_baseline.predict(X_test, verbose=0).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))
print("-" * 30)

# =================================================================
# LANGKAH 5: VISUALISASI LEARNING CURVE
# =================================================================
import matplotlib.pyplot as plt

print("--- Membuat Learning Curve ---")
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Learning Curve (Model Baseline)")
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve_baseline.png", dpi=120)
plt.show()

print("Learning curve berhasil disimpan sebagai learning_curve_baseline.png")