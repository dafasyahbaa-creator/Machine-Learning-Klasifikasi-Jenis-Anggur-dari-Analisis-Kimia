# UTS Intelegensi Buatan — Kasus 1: Klasifikasi Kultivar Anggur

Menentukan kultivar anggur (A, B, C) dari 13 hasil analisis kimia. Mencakup feature engineering, perbandingan model dengan pipeline scikit-learn (nested CV), evaluasi, penyimpanan joblib, aplikasi Streamlit, dan notebook Google Colab.

| | |
|---|---|
| Mata kuliah | Intelegensi Buatan (3 SKS), TIF RM 23 A |
| Dosen | Muhammad Ikhwan Fathulloh, S.Kom. |
| Nama / NIM | Muhammad Daffa Syahbaa F.P.N / 24552011326 |
| Dataset / target | `kasus01_wine.csv` / `kultivar` |

## Ringkasan hasil

| Item | Nilai |
|---|---|
| Model terpilih | SVM (RBF), `C=1`, `gamma=scale` (nested CV F1-macro 1,000 ± 0,000) |
| Varian fitur | C: 6 fitur rasio, tanpa `total_phenols` dan `flavanoids` (17 fitur) |
| Test (36 sampel) | Accuracy 1,000 · F1-macro 1,000 · MCC 1,000 · Log loss 0,059 |
| Baseline kelas mayoritas | 0,389 |

Catatan jujur: Wine adalah dataset kecil dan mudah. Dengan test 36 sampel, akurasi 100% memiliki selang kepercayaan 95% (Wilson) sekitar [0,90; 1,00], jadi bukan jaminan performa pada anggur lain.

## Kesesuaian dengan soal

| Tahap | Lokasi |
|---|---|
| 1. Feature engineering | `fitur.py` (audit, `FiturRasio`, pipeline), ablasi di `latih.py` |
| 2. Model klasifikasi (5 algoritma, pipeline) | `latih.py`: Logistic Regression, SVM, KNN, Random Forest, Gradient Boosting |
| 3. Metrik evaluasi | `models/metrics.json`, tab "Metrik evaluasi" di aplikasi, `Laporan/` di root repo |
| 4. Deploy joblib (pipeline + metadata) | `models/wine_kultivar.joblib` |
| 5. Aplikasi Streamlit | `app.py` |
| 6. Notebook Colab | `notebook/kasus01_colab.ipynb` |

| Aturan | Cara dipenuhi |
|---|---|
| Split sebelum imputasi/scaling/encoding | `train_test_split` stratified 80/20 dijalankan lebih dulu; imputer dan scaler di dalam `Pipeline` |
| Tentukan sendiri fitur, jelaskan yang dibuang | Semua 13 kolom sah dipakai (tidak ada ID/kolom pasca-kejadian); `total_phenols` dan `flavanoids` dibuang di varian C karena redundan dengan rasio |
| Test sekali; pilih model dengan CV di train | Ablasi fitur (CV 5×3) dan nested CV (loop dalam 3-fold, luar 5×2) hanya di train; test dievaluasi sekali |
| Model di `models/`, tanpa kredensial/path absolut | Semua path relatif (`Path(__file__)`) |

## Tugas khusus Kasus 1

- **Korelasi dan rasio.** Korelasi tertinggi `total_phenols`–`flavanoids` 0,86, `flavanoids`–`od280/od315` 0,79. Dibuat 6 rasio (flavonoid/fenol, nonflavonoid/fenol, proantosianin/fenol, od280/flavonoid, alkalinitas/abu, prolin/alkohol).
- **Buang fitur asli?** Diuji, bukan ditebak: tiga varian dibandingkan dengan CV di train (A asli saja 0,9838; B asli+rasio 0,9865; C 0,9887). C dipilih; selisihnya kecil dan di bawah simpangan baku CV, sehingga dilaporkan sebagai keputusan ringan.
- **Scaling.** `StandardScaler` di dalam pipeline (prolin ratusan–ribuan vs hue 0,5–1,7), dipelajari dari data train saja.

## Struktur

```
kasus01/
├─ app.py  fitur.py  latih.py  requirements.txt
├─ dataset/kasus01_wine.csv
├─ models/  wine_kultivar.joblib  metrics.json  laporan.md  figures/*.png
├─ notebook/kasus01_colab.ipynb
└─ .streamlit/config.toml
```

## Cara menjalankan

**Colab:** unggah notebook ke Colab → *Runtime → Run all* → buka tautan `trycloudflare.com` pada sel terakhir.

**Lokal:**
```
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python latih.py            # melatih ulang, menulis models/
streamlit run app.py
```
Semua proses acak memakai `random_state=42`. Bila versi scikit-learn Anda berbeda dari yang menyimpan file joblib, aplikasi melatih ulang otomatis.
