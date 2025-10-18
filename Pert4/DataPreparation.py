import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# --- Langkah 1: Baca dan siapkan data ---
df = pd.read_csv("../dataset/kelulusan_mahasiswa.csv")
df = df.drop_duplicates()

# Buat fitur baru
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
df.to_csv("processed_kelulusan.csv", index=False)

# --- Langkah 2: Fungsi untuk setiap visualisasi ---

def show_distribusi_ipk():
    plt.figure(figsize=(6, 4))
    sns.histplot(df['IPK'], bins=10, kde=True)
    plt.title("Distribusi IPK")
    plt.show()

def show_ipk_vs_belajar():
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
    plt.title("IPK vs Waktu Belajar")
    plt.show()

def show_boxplot_ipk():
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['IPK'])
    plt.title("Boxplot IPK")
    plt.show()

def show_heatmap_korelasi():
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Heatmap Korelasi")
    plt.show()

# --- Langkah 3: Buat GUI Tkinter ---
root = tk.Tk()
root.title("Visualisasi Data Kelulusan Mahasiswa")

label = tk.Label(root, text="Pilih Grafik yang Ingin Ditampilkan:", font=("Arial", 12, "bold"))
label.pack(pady=10)

btn1 = tk.Button(root, text="Distribusi IPK", command=show_distribusi_ipk, width=25)
btn1.pack(pady=5)

btn2 = tk.Button(root, text="IPK vs Waktu Belajar", command=show_ipk_vs_belajar, width=25)
btn2.pack(pady=5)

btn3 = tk.Button(root, text="Boxplot IPK", command=show_boxplot_ipk, width=25)
btn3.pack(pady=5)

btn4 = tk.Button(root, text="Heatmap Korelasi", command=show_heatmap_korelasi, width=25)
btn4.pack(pady=5)

root.mainloop()
