# ======================================================
# PROJECT MACHINE LEARNING: RANDOM FOREST (ALL VISUALS IN ONE FIGURE)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve
import joblib

# ====== Langkah 1: Muat Data ======
df = pd.read_csv("../dataset/processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

num_cols = X.select_dtypes(include="number").columns

# ====== Langkah 2: Pipeline + Model ======
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols)
], remainder="drop")

pipe = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42
    ))
])

# ====== Langkah 3: Validasi Silang ======
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)

# ====== Langkah 4: Grid Search ======
param = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=0)
gs.fit(X, y)

best_model = gs.best_estimator_
final_model = best_model
y_pred = final_model.predict(X)

f1_test = f1_score(y, y_pred, average="macro")
cm = confusion_matrix(y, y_pred)

# ====== Siapkan Semua Komponen Visual ======
# ROC
if hasattr(final_model, "predict_proba"):
    y_proba = final_model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = roc_auc_score(y, y_proba)
else:
    fpr, tpr, roc_auc = None, None, None

# Feature Importance
importances = final_model.named_steps["clf"].feature_importances_
fn = final_model.named_steps["pre"].get_feature_names_out()
feat_imp = pd.DataFrame({"Feature": fn, "Importance": importances}).sort_values(
    by="Importance", ascending=False
)

# ====== Langkah 5: Satu Figure Gabungan ======
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plt.suptitle("📊 Random Forest Model Overview", fontsize=14, fontweight="bold")

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0,0])
axes[0,0].set_title("Confusion Matrix")
axes[0,0].set_xlabel("Prediksi")
axes[0,0].set_ylabel("Aktual")

# Distribusi CV Score
sns.boxplot(scores, color="lightgreen", ax=axes[0,1])
axes[0,1].set_title("Distribusi F1 (Cross-Validation)")
axes[0,1].set_xlabel("F1-macro Score")
axes[0,1].grid(True, linestyle="--", alpha=0.6)

# ROC Curve
if fpr is not None:
    axes[1,0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[1,0].plot([0,1],[0,1],'--',color='gray')
    axes[1,0].set_title("ROC Curve")
    axes[1,0].set_xlabel("False Positive Rate")
    axes[1,0].set_ylabel("True Positive Rate")
    axes[1,0].legend()
else:
    axes[1,0].text(0.5,0.5,"ROC tidak tersedia", ha="center", va="center")

# Feature Importance
sns.barplot(data=feat_imp.head(10), x="Importance", y="Feature", palette="viridis", ax=axes[1,1])
axes[1,1].set_title("Top 10 Feature Importance")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ====== Langkah 6: Simpan Model ======
joblib.dump(final_model, "rf_model.pkl")
print("\n💾 Model disimpan sebagai rf_model.pkl")

# ====== Langkah 7: Inferensi Lokal ======
mdl = joblib.load("rf_model.pkl")
sample = pd.DataFrame([{
    "IPK": 3.4,
    "Jumlah_Absensi": 4,
    "Waktu_Belajar_Jam": 7,
    "Rasio_Absensi": 4/14,
    "IPK_x_Study": 3.4*7
}])

pred = int(mdl.predict(sample)[0])
print(f"\n🧠 Prediksi (sample): {pred}")
print("Best params:", gs.best_params_)
print(f"F1-score (final): {f1_test:.3f}")
