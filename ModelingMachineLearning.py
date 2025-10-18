# === PERTEMUAN 5 - MACHINE LEARNING PROJECT ===
# Nama: Farhan Nul Hakim
# NIM: 231011402153
# Kelas: 05TPLE015
# Dosen: Zurnan Alfian, S.Kom., M.Kom.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os

# === Fungsi untuk cek environment ===
def check_environment():
    print("Memeriksa environment...")
    try:
        import sklearn
        import numpy
        print(f"scikit-learn: {sklearn.__version__}, numpy: {numpy.__version__}")
    except Exception as e:
        print(f"Error memeriksa library: {str(e)}")
        sys.exit(1)

# Jalankan cek environment
check_environment()

# === Langkah 1: Muat dan Split Data ===
try:
    df = pd.read_csv("processed_kelulusan.csv")
    print("Data dimuat berhasil.")
except FileNotFoundError:
    print("Error: File 'processed_kelulusan.csv' tidak ditemukan.")
    sys.exit(1)

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42
)

# === Langkah 2: Baseline Model dan Pipeline ===
num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
], remainder="drop")

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

pipe_lr.fit(X_train, y_train)
y_val_pred = pipe_lr.predict(X_val)
print("Baseline (LogReg) F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# === Langkah 3: Model Alternatif (Random Forest) ===
rf = RandomForestClassifier(
    n_estimators=100, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])

pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)
print("RandomForest F1(val):", f1_score(y_val, y_val_rf, average="macro"))

# === Langkah 4: Validasi Silang & Tuning Ringkas ===
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
param = {
    "clf__max_depth": [None, 20],
    "clf__min_samples_split": [2, 5]
}

gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf, scoring="f1_macro", verbose=1)
print("Starting GridSearchCV...")
gs.fit(X_train, y_train)
print("GridSearchCV completed!")
print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)
print("Best RF F1(val):", f1_score(y_val, y_val_best, average="macro"))

# === Langkah 5: Evaluasi Akhir (Test Set) ===
final_model = best_rf
y_test_pred = final_model.predict(X_test)

print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# === FIGURE 1: Confusion Matrix Visual ===
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()
print("✅ Gambar 'confusion_matrix.png' berhasil dibuat.")

# === FIGURE 2: Perbandingan F1 Score Antar Model ===
models = ['Logistic Regression', 'Random Forest', 'Best RF']
scores = [
    f1_score(y_val, y_val_pred, average='macro'),
    f1_score(y_val, y_val_rf, average='macro'),
    f1_score(y_val, y_val_best, average='macro')
]

plt.figure(figsize=(6, 4))
sns.barplot(x=models, y=scores, palette='viridis')
plt.title("Perbandingan F1 Score Antar Model (Validation Set)")
plt.ylabel("F1 Macro Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("f1_comparison.png", dpi=120)
plt.close()
print("✅ Gambar 'f1_comparison.png' berhasil dibuat.")

# === FIGURE 3: ROC Curve (diperindah) ===
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_test_proba):.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve_pretty.png", dpi=120)
    plt.close()
    print("✅ Gambar 'roc_curve_pretty.png' berhasil dibuat.")

# === Langkah 6: Simpan Model ===
try:
    if not os.path.exists("model.pkl") or os.path.getsize("model.pkl") == 0:
        joblib.dump(final_model, "model.pkl")
        print("Model tersimpan ke model.pkl")
    else:
        print("File model.pkl sudah ada, model tidak disimpan ulang.")
except PermissionError:
    print("Error: Tidak ada izin untuk menyimpan model.pkl. Cek izin folder.")
except Exception as e:
    print(f"Error menyimpan model: {str(e)}")

print("\n✅ Semua proses selesai tanpa error.")
print("File gambar dan model telah disimpan di folder proyek.")
