# UTS Intelegensi Buatan – Kasus 1: Klasifikasi Kultivar Anggur

Klasifikasi **kultivar anggur** (Kultivar A, B, C) dari 13 hasil analisis kimia. Proyek ini mencakup feature engineering, perbandingan model dengan pipeline scikit-learn (nested CV), evaluasi, penyimpanan model dengan joblib, aplikasi Streamlit, dan notebook Google Colab.

|                      |                                                                              |
| -------------------- | ---------------------------------------------------------------------------- |
| **Mata kuliah**      | Intelegensi Buatan (3 SKS)                                                   |
| **Dosen pengampu**   | Muhammad Ikhwan Fathulloh, S.Kom.                                            |
| **Nama**             | Muhammad Daffa Syahbaa F.P.N                                                 |
| **NIM**              | 24552011326                                                                  |
| **Kelas**            | TIF RM 23 A                                                                  |
| **Program studi**    | Teknik Informatika, Fakultas Industri Kreatif, Universitas Teknologi Bandung |
| **Kasus**            | Kasus 1 (jumlah digit NIM = 31, 31 mod 10 = 1)                               |
| **Dataset / target** | `kasus01_wine.csv` / `kultivar`                                              |

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dafasyahbaa-creator/Machine-Learning-Klasifikasi-Jenis-Anggur-dari-Analisis-Kimia/blob/main/kasus01/notebook/kasus01_colab.ipynb)

---

## Daftar isi

1. [Ringkasan hasil](#1-ringkasan-hasil)
2. [Kesesuaian dengan ketentuan UTS](#2-kesesuaian-dengan-ketentuan-uts)
3. [Dataset](#3-dataset)
4. [Feature engineering](#4-feature-engineering)
5. [Model dan evaluasi](#5-model-dan-evaluasi)
6. [Aplikasi Streamlit](#6-aplikasi-streamlit)
7. [Struktur repository](#7-struktur-repository)
8. [Cara menjalankan](#8-cara-menjalankan)
9. [Keputusan penting dan alasannya](#9-keputusan-penting-dan-alasannya)
10. [Keterbatasan](#10-keterbatasan)

---

## 1. Ringkasan hasil

| Item                       | Nilai                                                           |
| -------------------------- | --------------------------------------------------------------- |
| Model terpilih             | **SVM (RBF)** (nested CV F1-macro tertinggi: 1,000 ± 0,000)     |
| Varian fitur               | **C**: 6 fitur rasio, tanpa `total_phenols` dan `flavanoids` (17 fitur) |
| Accuracy (test, 36 data)   | **1,000** (baseline kelas mayoritas 0,389; selang kepercayaan 95% [0,904; 1,000]) |
| F1-macro (metrik utama)    | **1,000**                                                       |
| F1-weighted                | 1,000                                                           |
| Balanced accuracy          | 1,000                                                           |
| MCC                        | 1,000                                                           |
| Log loss                   | 0,059                                                           |

Seleksi fitur dan model memakai cross-validation **hanya di data train**; data test dievaluasi **satu kali**. Wine adalah dataset kecil dan mudah, jadi skor sempurna bukan jaminan performa pada anggur lain (lihat [Keterbatasan](#10-keterbatasan)).

## 2. Kesesuaian dengan ketentuan UTS

**Enam tahapan wajib**

| # | Tahapan                                              | Lokasi                                                                                                   |
| - | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1 | Feature engineering (audit, fitur baru, ubah fitur)  | `kasus01/fitur.py`, ablasi di `kasus01/latih.py`, `Laporan/`                                             |
| 2 | Model klasifikasi (minimal dua algoritma, pipeline)  | `kasus01/latih.py` (Logistic Regression, SVM, KNN, Random Forest, Gradient Boosting)                     |
| 3 | Metrik evaluasi dan maknanya                         | `kasus01/models/metrics.json`, `kasus01/models/laporan.md`, `Laporan/`                                   |
| 4 | Deploy dengan joblib (pipeline + metadata)           | `kasus01/models/wine_kultivar.joblib`                                                                    |
| 5 | Aplikasi Streamlit                                   | `kasus01/app.py`                                                                                         |
| 6 | Notebook Google Colab                                | `kasus01/notebook/kasus01_colab.ipynb`                                                                   |

**Aturan yang wajib dipatuhi**

| Aturan                                                        | Cara dipenuhi                                                                                                      |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Split sebelum imputasi, scaling, encoding                     | `train_test_split` stratified 80/20 dijalankan lebih dulu; imputer dan scaler ada di dalam `Pipeline`              |
| Tentukan sendiri kolom fitur dan alasan membuang kolom        | Ke-13 kolom layak dipakai (tidak ada ID atau kolom pasca-kejadian); `total_phenols` dan `flavanoids` dibuang di varian C karena redundan dengan rasio |
| Evaluasi test satu kali, pilih model dengan CV di train       | Ablasi fitur (CV 5×3) dan nested CV (dalam 3-fold, luar 5×2) hanya di train; test dipakai sekali                   |
| Model di folder `models/`, tanpa kredensial atau path absolut | Semua path relatif (`Path(__file__)`); tidak ada kredensial                                                        |

**File yang dikumpulkan:** `fitur.py`, `latih.py`, `models/*.joblib`, `app.py`, notebook Colab, dan laporan singkat (folder `Laporan/`).

**Tugas khusus Kasus 1**

- **Periksa korelasi, buat rasio, putuskan membuang fitur asli:** bagian 4 dan 9.
- **Scaling yang benar:** `StandardScaler` di dalam pipeline, dipelajari dari data train saja (bagian 4).

## 3. Dataset

Sumber: folder `UTS/dataset` pada repository mata kuliah (`Artificial-Intelegent-Course-Bank`). Data berisi 178 sampel dengan 13 fitur numerik dan 1 target.

| Kolom                          | Keterangan                          | Dipakai model                          |
| ------------------------------ | ----------------------------------- | -------------------------------------- |
| `alcohol`                      | Kadar alkohol                       | Ya                                     |
| `malic_acid`                   | Asam malat                          | Ya                                     |
| `ash`                          | Kadar abu                           | Ya                                     |
| `alcalinity_of_ash`            | Alkalinitas abu                     | Ya                                     |
| `magnesium`                    | Kadar magnesium                     | Ya                                     |
| `total_phenols`                | Total fenol                         | Hanya lewat rasio (dibuang di varian C) |
| `flavanoids`                   | Flavonoid                           | Hanya lewat rasio (dibuang di varian C) |
| `nonflavanoid_phenols`         | Fenol nonflavonoid                  | Ya                                     |
| `proanthocyanins`              | Proantosianin                       | Ya                                     |
| `color_intensity`              | Intensitas warna                    | Ya                                     |
| `hue`                          | Rona warna                          | Ya                                     |
| `od280_od315_of_diluted_wines` | Rasio absorbansi anggur terencerkan | Ya                                     |
| `proline`                      | Kadar prolin                        | Ya                                     |
| `kultivar`                     | **Target**: Kultivar A, B, C        | Target                                 |

**Distribusi kelas:** Kultivar B 71 (39,9%), Kultivar A 59 (33,1%), Kultivar C 48 (27,0%).

**Hasil audit data**

- Tidak ada baris duplikat dan tidak ada nilai kosong.
- Tipe data: 13 kolom `float64` dan target bertipe teks (kategorik).
- Tidak ada nilai ≤ 0 yang tidak wajar.
- Outlier IQR ringan: `malic_acid` 3, `ash` 3, `alcalinity_of_ash` 4, `magnesium` 4, `proanthocyanins` 2, `color_intensity` 4, `hue` 1. Nilainya masuk akal secara kimia sehingga dipertahankan.

## 4. Feature engineering

Semua langkah ada di `fitur.py` dan berada **di dalam Pipeline**, sehingga statistik preprocessing hanya dipelajari dari data train.

**Korelasi antarfitur (tertinggi)**

| Pasangan fitur                               | Korelasi |
| -------------------------------------------- | -------- |
| `total_phenols` – `flavanoids`               | 0,865    |
| `flavanoids` – `od280_od315_of_diluted_wines`| 0,787    |
| `total_phenols` – `od280_od315_of_diluted_wines` | 0,700 |
| `flavanoids` – `proanthocyanins`             | 0,653    |
| `alcohol` – `proline`                        | 0,644    |

**Fitur baru (6 rasio dari pasangan fitur yang saling terkait)**

| Fitur                       | Rumus                                |
| --------------------------- | ------------------------------------ |
| `rasio_flavonoid_fenol`     | `flavanoids` / `total_phenols`       |
| `rasio_nonflavonoid_fenol`  | `nonflavanoid_phenols` / `total_phenols` |
| `rasio_proantosianin_fenol` | `proanthocyanins` / `total_phenols`  |
| `rasio_od_flavonoid`        | `od280_od315_of_diluted_wines` / `flavanoids` |
| `rasio_alkalinitas_abu`     | `alcalinity_of_ash` / `ash`          |
| `rasio_prolin_alkohol`      | `proline` / `alcohol`                |

**Apakah fitur asli perlu dibuang?** Diuji dengan CV di data train (F1-macro, 5 fold × 3 ulangan):

| Varian fitur                                    | Logistic Regression | SVM (RBF) | Random Forest | Rata-rata |
| ----------------------------------------------- | ------------------- | --------- | ------------- | --------- |
| A. Fitur asli saja (13)                         | 0,9815              | 0,9886    | 0,9812        | 0,9838    |
| B. Asli + rasio (19)                            | 0,9867              | 0,9959    | 0,9767        | 0,9865    |
| **C. Rasio, tanpa `total_phenols` & `flavanoids` (17)** | 0,9891      | 0,9979    | 0,9790        | **0,9887** |

Varian C dipilih. Selisih antarvarian kecil (di bawah simpangan baku CV), jadi keputusan ini dilaporkan apa adanya: tidak lebih buruk dan mengurangi redundansi, tetapi tidak diklaim terbukti lebih baik secara statistik.

**Pipeline preprocessing**

```
imputasi median (data train) -> fitur rasio -> StandardScaler -> model
```

Scaling wajib karena skala fitur sangat berbeda (`proline` 278–1680 vs `hue` 0,48–1,71). Scaler dipasang di semua model agar perbandingan adil.

## 5. Model dan evaluasi

Split stratified 80/20 (`random_state=42`): **142 train, 36 test**.

**Perbandingan model (nested cross-validation di data train)**

| Model                    | CV F1-macro       | CV Accuracy | Balanced accuracy |
| ------------------------ | ----------------- | ----------- | ----------------- |
| **SVM (RBF)** (terpilih) | **1,000 ± 0,000** | 1,000       | 1,000             |
| Logistic Regression      | 0,986 ± 0,023     | 0,986       | 0,989             |
| Random Forest            | 0,979 ± 0,028     | 0,979       | 0,982             |
| KNN                      | 0,967 ± 0,022     | 0,965       | 0,971             |
| Gradient Boosting        | 0,956 ± 0,050     | 0,955       | 0,959             |

Hiperparameter SVM terpilih lewat `GridSearchCV` di data train: `C=1.0`, `gamma="scale"`.

**Fitur terpenting (importansi permutasi, data train):** `flavanoids` (0,205), `total_phenols` (0,046), `proline` (0,045), `alcalinity_of_ash` (0,032), `proanthocyanins` (0,031).

**Hasil pada data test (satu kali)**

| Metrik                | Nilai |
| --------------------- | ----- |
| Accuracy              | 1,000 |
| Balanced accuracy     | 1,000 |
| F1-macro              | 1,000 |
| F1-weighted           | 1,000 |
| MCC                   | 1,000 |
| Log loss              | 0,059 |
| ROC-AUC (OvR, macro)  | 1,000 |

| Kelas      | Precision | Recall | F1    | Support |
| ---------- | --------- | ------ | ----- | ------- |
| Kultivar A | 1,00      | 1,00   | 1,00  | 12      |
| Kultivar B | 1,00      | 1,00   | 1,00  | 14      |
| Kultivar C | 1,00      | 1,00   | 1,00  | 10      |

**Confusion matrix** (baris = aktual, kolom = prediksi)

|                | Pred A | Pred B | Pred C |
| -------------- | ------ | ------ | ------ |
| **Kultivar A** | 12     | 0      | 0      |
| **Kultivar B** | 0      | 14     | 0      |
| **Kultivar C** | 0      | 0      | 10     |

**Makna metrik:** *accuracy* adalah proporsi tebakan benar. *F1-macro* merata-ratakan F1 tiap kelas sehingga kelas kecil sama pentingnya. *MCC* adalah korelasi prediksi dan label yang memperhitungkan seluruh isi confusion matrix (1 = sempurna). *Log loss* menilai kualitas probabilitas, bukan hanya label (makin kecil makin baik).

**Interpretasi:** semua 36 sampel uji diprediksi benar dan jauh di atas baseline 0,389. Log loss 0,059 menunjukkan model juga yakin pada prediksinya. Karena sampel uji hanya 36, selang kepercayaan 95% untuk accuracy masih lebar ([0,904; 1,000]).

## 6. Aplikasi Streamlit

`app.py` menerima 13 nilai mentah hasil analisis kimia, menampilkan prediksi beserta probabilitas, dan metrik evaluasi.

| Tab                    | Isi                                                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Prediksi**           | Form input 13 nilai (bisa diisi cepat dengan contoh Kultivar A/B/C), hasil prediksi, grafik dan tabel probabilitas, peringatan bila nilai di luar rentang data latih |
| **Metrik evaluasi**    | Accuracy, balanced accuracy, F1-macro, MCC, log loss, confusion matrix, laporan per kelas, perbandingan algoritma, ablasi fitur, importansi fitur |
| **Prediksi batch (CSV)** | Unggah CSV berisi banyak sampel, hasilnya bisa diunduh                                                                                  |
| **Tentang**            | Catatan keputusan dan keterbatasan                                                                                                        |

Konfigurasi `.streamlit/config.toml` menonaktifkan prompt email saat pertama kali menjalankan Streamlit.

## 7. Struktur repository

```
.
├─ README.md
├─ .gitignore
├─ Laporan/
│  ├─ Laporan_UTS_Kasus1_24552011326_M_DaffaS_F_P_N.docx
│  └─ Laporan_UTS_Kasus1_24552011326_M_DaffaS_F_P_N.pdf
└─ kasus01/
   ├─ app.py                   aplikasi Streamlit
   ├─ fitur.py                 feature engineering
   ├─ latih.py                 pelatihan, evaluasi, simpan model
   ├─ requirements.txt
   ├─ README.md
   ├─ dataset/
   │  └─ kasus01_wine.csv
   ├─ models/
   │  ├─ wine_kultivar.joblib  pipeline + metadata
   │  ├─ metrics.json          metrik hasil run
   │  ├─ laporan.md            ringkasan otomatis
   │  └─ figures/              grafik (korelasi, PCA, confusion matrix, ...)
   ├─ notebook/
   │  └─ kasus01_colab.ipynb
   └─ .streamlit/
      └─ config.toml
```

## 8. Cara menjalankan

### A. Google Colab (disarankan)

1. Buka notebook lewat tombol **Buka di Colab** di atas.
2. Pilih *Runtime → Run all*, atau jalankan sel dari atas ke bawah.
3. Sel terakhir mencetak tautan aplikasi Streamlit; buka tautan itu.

### B. Lokal

```
git clone https://github.com/dafasyahbaa-creator/Machine-Learning-Klasifikasi-Jenis-Anggur-dari-Analisis-Kimia.git
cd Machine-Learning-Klasifikasi-Jenis-Anggur-dari-Analisis-Kimia/kasus01

python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/Mac: source .venv/bin/activate

python -m pip install -r requirements.txt
python latih.py            # opsional: melatih ulang dan menulis ulang models/
streamlit run app.py       # buka http://localhost:8501
```

**Reproduksibilitas:** seluruh proses acak memakai `random_state=42`, sehingga `python latih.py` menghasilkan metrik yang sama.

## 9. Keputusan penting dan alasannya

**Split sebelum preprocessing.** Data dibagi 80/20 stratified lebih dulu. Imputasi dan scaling ada di dalam `Pipeline` sehingga hanya belajar dari data train, dan data test tidak bocor ke proses pelatihan.

**`total_phenols` dan `flavanoids` dibuang di varian C.** Keduanya berkorelasi 0,865 dan informasinya sudah terwakili rasio berbasis fenol. Keputusan ini diuji dengan CV, bukan ditebak. Selisih skornya kecil, jadi tidak diklaim terbukti lebih baik.

**SVM (RBF) dipilih lewat nested CV.** Hiperparameter disetel di lipatan dalam, performa diestimasi di lipatan luar, sehingga estimasi tidak terlalu optimis. SVM unggul dan paling stabil antar fold (std 0).

**Scaling di semua model.** Skala fitur sangat berbeda (prolin ratusan hingga ribuan, hue 0,5–1,7). SVM dan KNN sensitif terhadap skala; scaler yang sama dipasang di semua model agar perbandingan adil.

**Metrik utama F1-macro.** Kelas cukup seimbang tetapi tidak persis sama (71/59/48), sehingga F1-macro dipakai agar setiap kultivar sama pentingnya. Accuracy, MCC, dan log loss dilaporkan sebagai pendamping.

## 10. Keterbatasan

- Test hanya 36 data; satu kesalahan menggeser accuracy sekitar 2,8 poin, dan selang kepercayaan 95% untuk akurasi 100% adalah sekitar [0,90; 1,00].
- Wine adalah dataset kecil dan mudah, sehingga skor sempurna tidak boleh dibaca sebagai jaminan performa pada anggur lain.
- Model hanya berlaku untuk tiga kultivar dari satu wilayah yang sama.
- Selisih antarvarian fitur kecil (di bawah simpangan baku CV), sehingga pilihan varian C bersifat ringan, bukan bukti kuat.
