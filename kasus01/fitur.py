"""
fitur.py - Feature engineering untuk Kasus 1: Klasifikasi kultivar anggur.

Isi modul:
  * muat_data()      : membaca dataset (lokal -> unduh GitHub -> cadangan scikit-learn)
  * bersihkan()      : pembersihan dasar (duplikat, target kosong, nilai tidak wajar)
  * audit_data()     : audit data (duplikat, missing, tipe data, outlier, korelasi)
  * FiturRasio       : transformer scikit-learn untuk fitur rasio (stateless, aman dari
                       kebocoran data karena tidak menghitung statistik apa pun)
  * buat_pipeline()  : pipeline lengkap  imputasi -> fitur rasio -> scaling -> model

Semua statistik preprocessing (median imputasi, mean/std scaler) dipelajari HANYA dari
data train karena berada di dalam Pipeline yang di-fit pada data train.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
NAMA_FILE = "kasus01_wine.csv"
TARGET = "kultivar"

URL_DATASET = [
    "https://raw.githubusercontent.com/Muhammad-Ikhwan-Fathulloh/"
    "Artificial-Intelegent-Course-Bank/main/UTS/dataset/" + NAMA_FILE,
    "https://raw.githubusercontent.com/Muhammad-Ikhwan-Fathulloh/"
    "ArtificialIntelegent-Course-Bank/main/UTS/dataset/" + NAMA_FILE,
]

# 13 komponen kimia anggur (fitur mentah) beserta metadata untuk antarmuka aplikasi
INFO_FITUR = {
    "alcohol": ("Alkohol", "% vol", "Kadar alkohol", 0.01, "%.2f"),
    "malic_acid": ("Asam malat", "g/L", "Kadar asam malat", 0.01, "%.2f"),
    "ash": ("Abu (ash)", "g/L", "Kadar abu", 0.01, "%.2f"),
    "alcalinity_of_ash": ("Alkalinitas abu", "", "Alkalinitas abu", 0.1, "%.1f"),
    "magnesium": ("Magnesium", "mg/L", "Kadar magnesium", 1.0, "%.0f"),
    "total_phenols": ("Total fenol", "", "Total komponen fenolik", 0.01, "%.2f"),
    "flavanoids": ("Flavonoid", "", "Fenol jenis flavonoid", 0.01, "%.2f"),
    "nonflavanoid_phenols": ("Fenol nonflavonoid", "", "Fenol non-flavonoid", 0.01, "%.2f"),
    "proanthocyanins": ("Proantosianin", "", "Proantosianin", 0.01, "%.2f"),
    "color_intensity": ("Intensitas warna", "", "Intensitas warna anggur", 0.01, "%.2f"),
    "hue": ("Hue (rona)", "", "Rona warna anggur", 0.01, "%.2f"),
    "od280_od315_of_diluted_wines": (
        "OD280/OD315", "", "Rasio absorbansi anggur terencerkan", 0.01, "%.2f"),
    "proline": ("Prolin", "mg/L", "Kadar prolin", 1.0, "%.0f"),
}
FITUR_ASLI = list(INFO_FITUR.keys())

# Fitur rasio: pasangan fitur yang saling terkait secara kimia
# nama_fitur_baru: (pembilang, penyebut, alasan)
RASIO = {
    "rasio_flavonoid_fenol": ("flavanoids", "total_phenols",
                              "porsi flavonoid dalam total fenol"),
    "rasio_nonflavonoid_fenol": ("nonflavanoid_phenols", "total_phenols",
                                 "porsi fenol nonflavonoid dalam total fenol"),
    "rasio_proantosianin_fenol": ("proanthocyanins", "total_phenols",
                                  "porsi proantosianin dalam total fenol"),
    "rasio_od_flavonoid": ("od280_od315_of_diluted_wines", "flavanoids",
                           "absorbansi per satuan flavonoid"),
    "rasio_alkalinitas_abu": ("alcalinity_of_ash", "ash",
                              "alkalinitas relatif terhadap kadar abu"),
    "rasio_prolin_alkohol": ("proline", "alcohol",
                             "prolin relatif terhadap kadar alkohol"),
}

# Varian fitur yang dibandingkan pada tahap ablasi (dipilih dengan CV di data train)
VARIAN_FITUR = {
    "A. Fitur asli saja (13)": dict(tambah_rasio=False, buang=()),
    "B. Asli + rasio (19)": dict(tambah_rasio=True, buang=()),
    "C. Rasio, tanpa total_phenols & flavanoids (17)": dict(
        tambah_rasio=True, buang=("total_phenols", "flavanoids")),
}


# ----------------------------------------------------------------------------------
# Pemuatan dan pembersihan data
# ----------------------------------------------------------------------------------
def normalisasi_kolom(nama: str) -> str:
    """Samakan format nama kolom: huruf kecil, spasi dan '/' menjadi '_'."""
    nama = str(nama).strip().lower()
    nama = re.sub(r"[\s/\-\.]+", "_", nama)
    return re.sub(r"_+", "_", nama).strip("_")


def _rapikan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalisasi_kolom(c) for c in df.columns]
    # buang kolom indeks bawaan jika ada
    df = df.drop(columns=[c for c in df.columns if c.startswith("unnamed")], errors="ignore")
    if TARGET not in df.columns:
        for alt in ("target", "class", "cultivar", "kelas"):
            if alt in df.columns:
                df = df.rename(columns={alt: TARGET})
                break
    if TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
        nilai = sorted(df[TARGET].dropna().unique())
        huruf = "ABC"
        df[TARGET] = df[TARGET].map({v: f"Kultivar {huruf[i]}" for i, v in enumerate(nilai)})
    elif TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(str).str.strip()
    return df


def _cadangan_sklearn() -> pd.DataFrame:
    from sklearn.datasets import load_wine

    d = load_wine(as_frame=True).frame
    d["target"] = d["target"].map({0: "Kultivar A", 1: "Kultivar B", 2: "Kultivar C"})
    return d.rename(columns={"target": TARGET})


def muat_data(path: str | Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Baca dataset. Urutan: path eksplisit -> dataset/ -> unduh GitHub -> cadangan sklearn."""
    kandidat = [Path(path)] if path else []
    kandidat += [BASE_DIR / "dataset" / NAMA_FILE, Path("dataset") / NAMA_FILE, Path(NAMA_FILE)]
    for p in kandidat:
        if p.exists():
            if verbose:
                print(f"[data] membaca {p.name} dari {p.parent.name or '.'}/")
            return _rapikan(pd.read_csv(p))
    tujuan = BASE_DIR / "dataset" / NAMA_FILE
    for url in URL_DATASET:
        try:
            tujuan.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, tujuan)
            if verbose:
                print("[data] berhasil diunduh dari repositori mata kuliah")
            return _rapikan(pd.read_csv(tujuan))
        except Exception:  # noqa: BLE001 - jaringan bisa gagal dengan berbagai cara
            continue
    if verbose:
        print("[data] PERINGATAN: file tidak ditemukan; memakai Wine dataset dari scikit-learn "
              "(data UCI yang sama, 178 sampel).")
    return _rapikan(_cadangan_sklearn())


def bersihkan(df: pd.DataFrame) -> pd.DataFrame:
    """Pembersihan dasar yang tidak menghitung statistik apa pun (aman sebelum split).

    * hapus baris duplikat dan baris tanpa target
    * nilai <= 0 pada fitur kimia dianggap tidak wajar -> NaN (diimputasi di dalam pipeline)
    """
    df = _rapikan(df).drop_duplicates().dropna(subset=[TARGET]).reset_index(drop=True)
    for kol in FITUR_ASLI:
        df[kol] = pd.to_numeric(df[kol], errors="coerce")
        df.loc[df[kol] <= 0, kol] = np.nan
    return df[FITUR_ASLI + [TARGET]]


def audit_data(df: pd.DataFrame, ambang_korelasi: float = 0.6) -> dict:
    """Audit data: bentuk, duplikat, missing, tipe, nilai tidak wajar, outlier IQR, korelasi."""
    df = _rapikan(df)
    fitur = df[FITUR_ASLI]
    q1, q3 = fitur.quantile(0.25), fitur.quantile(0.75)
    iqr = q3 - q1
    outlier = ((fitur < q1 - 1.5 * iqr) | (fitur > q3 + 1.5 * iqr)).sum()
    kor = fitur.corr()
    pasangan = (
        kor.where(np.triu(np.ones(kor.shape), 1).astype(bool))
        .stack()
        .rename("korelasi")
        .reset_index()
        .rename(columns={"level_0": "fitur_1", "level_1": "fitur_2"})
    )
    tinggi = pasangan[pasangan["korelasi"].abs() >= ambang_korelasi].sort_values(
        "korelasi", key=abs, ascending=False)
    return {
        "jumlah_baris": int(len(df)),
        "jumlah_kolom": int(df.shape[1]),
        "duplikat": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "missing_per_kolom": df.isna().sum()[lambda s: s > 0].to_dict(),
        "tipe_data": {k: str(v) for k, v in df.dtypes.items()},
        "nilai_tidak_wajar_<=0": int((fitur <= 0).sum().sum()),
        "outlier_iqr": outlier[outlier > 0].to_dict(),
        "distribusi_kelas": df[TARGET].value_counts().to_dict(),
        "korelasi_tinggi": tinggi.round(3).to_dict("records"),
        "statistik": df[FITUR_ASLI].describe().T.round(3),
    }


# ----------------------------------------------------------------------------------
# Transformer fitur rasio
# ----------------------------------------------------------------------------------
class FiturRasio(BaseEstimator, TransformerMixin):
    """Menambah fitur rasio dan (opsional) membuang kolom asli.

    Transformer ini *stateless*: rasio dihitung baris demi baris sehingga tidak ada
    informasi dari data test yang masuk ke data train (tidak ada kebocoran data).
    """

    def __init__(self, tambah_rasio: bool = True, buang: tuple = ()):
        self.tambah_rasio = tambah_rasio
        self.buang = buang

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(
            list(X.columns) if isinstance(X, pd.DataFrame) else FITUR_ASLI)
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=self.feature_names_in_)
        if self.tambah_rasio:
            for nama, (atas, bawah, _) in RASIO.items():
                df[nama] = df[atas] / df[bawah].clip(lower=1e-9)
        if self.buang:
            df = df.drop(columns=list(self.buang))
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.transform(pd.DataFrame(
            np.ones((1, len(self.feature_names_in_))), columns=self.feature_names_in_)).columns)


# ----------------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------------
def buat_model(nama: str):
    """Daftar algoritma yang dibandingkan."""
    if nama == "Logistic Regression":
        return LogisticRegression(C=1.0, max_iter=5000, random_state=RANDOM_STATE)
    if nama == "SVM (RBF)":
        return SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE)
    if nama == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    if nama == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    if nama == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    raise ValueError(f"Model tidak dikenal: {nama}")


NAMA_MODEL = ["Logistic Regression", "SVM (RBF)", "KNN", "Random Forest", "Gradient Boosting"]


def buat_pipeline(nama_model: str, tambah_rasio: bool = True, buang: tuple = ()) -> Pipeline:
    """imputasi median -> fitur rasio -> StandardScaler -> model.

    Scaler dipasang pada SEMUA model agar perbandingan konsisten; untuk model berbasis
    pohon scaling tidak mengubah hasil, tetapi untuk LR/SVM/KNN scaling wajib karena
    skala fitur sangat berbeda (proline ~ ratusan-ribuan, hue ~ 0,5-1,7).
    """
    imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
    return Pipeline([
        ("imputasi", imputer),
        ("fitur", FiturRasio(tambah_rasio=tambah_rasio, buang=tuple(buang))),
        ("scaling", StandardScaler()),
        ("model", buat_model(nama_model)),
    ])


GRID_TUNING = {
    "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
    "SVM (RBF)": {"model__C": [1.0, 10.0], "model__gamma": ["scale", 0.01]},
    "KNN": {"model__n_neighbors": [3, 5, 9], "model__weights": ["uniform", "distance"]},
    "Random Forest": {"model__max_depth": [None, 5], "model__min_samples_leaf": [1, 2]},
    "Gradient Boosting": {"model__n_estimators": [100, 200], "model__max_depth": [2, 3]},
}
