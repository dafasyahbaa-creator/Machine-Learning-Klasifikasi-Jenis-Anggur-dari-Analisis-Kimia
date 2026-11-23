"""
latih.py - Pelatihan, evaluasi, dan penyimpanan model Kasus 1 (kultivar anggur).

Alur (sesuai aturan UTS):
  1. Muat data, audit, bersihkan (tanpa menghitung statistik apa pun).
  2. SPLIT train/test (stratified 80/20) SEBELUM imputasi, scaling, atau encoding.
  3. Pilih varian fitur dan algoritma dengan cross-validation HANYA di data train
     (ablasi fitur + nested CV: tuning di loop dalam, estimasi di loop luar).
  4. Latih model terpilih pada seluruh data train, evaluasi pada data test SATU KALI.
  5. Simpan pipeline + metadata ke models/wine_kultivar.joblib dan metrik ke JSON.

Jalankan:  python latih.py
"""
from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, train_test_split)

import fitur as F

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DIR_MODEL = BASE_DIR / "models"
DIR_FIGUR = DIR_MODEL / "figures"
FILE_MODEL = DIR_MODEL / "wine_kultivar.joblib"
TEST_SIZE = 0.2
RS = F.RANDOM_STATE


def wilson(benar: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Interval kepercayaan 95% (Wilson) untuk akurasi."""
    p = benar / n
    pusat = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    lebar = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(pusat - lebar), float(min(1.0, pusat + lebar))


def hitung_metrik(y_true, y_pred, proba, kelas) -> dict:
    """Metrik klasifikasi multikelas pada data test."""
    cm = confusion_matrix(y_true, y_pred, labels=kelas)
    lo, hi = wilson(int((np.array(y_true) == np.array(y_pred)).sum()), len(y_true))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "accuracy_ci95": [lo, hi],
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, proba, labels=kelas),
        "roc_auc_ovr_macro": roc_auc_score(y_true, proba, multi_class="ovr",
                                           average="macro", labels=kelas),
        "confusion_matrix": cm.tolist(),
        "laporan_kelas": classification_report(y_true, y_pred, labels=kelas,
                                               output_dict=True, zero_division=0),
        "n_test": int(len(y_true)),
        "n_salah": int((np.array(y_true) != np.array(y_pred)).sum()),
    }


# ----------------------------------------------------------------------------------
# Visualisasi (disimpan ke models/figures untuk laporan dan aplikasi)
# ----------------------------------------------------------------------------------
def buat_gambar(Xtr, ytr, Xte, yte, ypred, kelas, hasil_cv, importansi, cm):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    DIR_FIGUR.mkdir(parents=True, exist_ok=True)
    warna = {"Kultivar A": "#2E86AB", "Kultivar B": "#F6AE2D", "Kultivar C": "#C44536"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1. distribusi kelas
    fig, ax = plt.subplots(figsize=(4.2, 3))
    jumlah = ytr.value_counts().reindex(kelas)
    ax.bar(kelas, jumlah.values, color=[warna[k] for k in kelas])
    for i, v in enumerate(jumlah.values):
        ax.text(i, v + 0.8, str(v), ha="center")
    ax.set_title("Distribusi kelas (data train)")
    ax.set_ylabel("Jumlah sampel")
    fig.tight_layout()
    fig.savefig(DIR_FIGUR / "distribusi_kelas.png", dpi=160)
    plt.close(fig)

    # 2. korelasi (data train)
    kor = Xtr.corr()
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(kor, cmap="RdBu_r", vmin=-1, vmax=1)
    nama = [c.replace("od280_od315_of_diluted_wines", "od280/od315") for c in kor.columns]
    ax.set_xticks(range(len(nama)), nama, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(nama)), nama, fontsize=8)
    for i in range(len(nama)):
        for j in range(len(nama)):
            if abs(kor.iat[i, j]) >= 0.6 and i != j:
                ax.text(j, i, f"{kor.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Korelasi Pearson")
    ax.set_title("Korelasi antarfitur (data train)")
    fig.tight_layout()
    fig.savefig(DIR_FIGUR / "korelasi.png", dpi=160)
    plt.close(fig)

    # 3. perbandingan model (nested CV)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    h = hasil_cv.sort_values("f1_macro_mean")
    ax.barh(h["model"], h["f1_macro_mean"], xerr=h["f1_macro_std"], color="#2E86AB",
            capsize=3)
    batas = max(0.85, h["f1_macro_mean"].min() - 0.07)
    ax.set_xlim(batas, 1.01)
    for i, v in enumerate(h["f1_macro_mean"]):
        ax.text(batas + 0.003, i, f"{v:.3f}", va="center", ha="left", color="white",
                fontsize=9, fontweight="bold")
    ax.set_xlabel("F1-macro (nested CV, data train)")
    ax.set_title("Perbandingan algoritma")
    fig.tight_layout()
    fig.savefig(DIR_FIGUR / "perbandingan_model.png", dpi=160)
    plt.close(fig)

    # 4. confusion matrix (test)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cm = np.array(cm)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(kelas)), [k.replace("Kultivar ", "") for k in kelas])
    ax.set_yticks(range(len(kelas)), [k.replace("Kultivar ", "") for k in kelas])
    for i in range(len(kelas)):
        for j in range(len(kelas)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion matrix (data test)")
    fig.tight_layout()
    fig.savefig(DIR_FIGUR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # 5. proyeksi PCA 2D (scaler dan PCA di-fit pada data train saja)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=2, random_state=RS).fit(sc.transform(Xtr))
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for k in kelas:
        a = pca.transform(sc.transform(Xtr[ytr == k]))
        ax.scatter(a[:, 0], a[:, 1], s=26, alpha=0.75, color=warna[k], label=f"{k} (train)")
        b = pca.transform(sc.transform(Xte[yte == k]))
        ax.scatter(b[:, 0], b[:, 1], s=60, marker="*", edgecolor="black", linewidth=0.5,
                   color=warna[k], label=f"{k} (test)")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} varians)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%} varians)")
    ax.set_title("Proyeksi PCA: tiga kultivar terpisah baik")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(DIR_FIGUR / "pca_2d.png", dpi=160)
    plt.close(fig)

    # 6. importansi permutasi
    fig, ax = plt.subplots(figsize=(5.4, 4))
    imp = importansi.sort_values("rata_rata")
    ax.barh(imp["fitur"], imp["rata_rata"], xerr=imp["std"], color="#6C757D", capsize=2)
    ax.set_xlabel("Penurunan F1-macro bila fitur diacak")
    ax.set_title("Importansi permutasi (data train)")
    fig.tight_layout()
    fig.savefig(DIR_FIGUR / "importansi.png", dpi=160)
    plt.close(fig)


def tulis_laporan_md(b: dict) -> None:
    """Ringkasan hasil otomatis -> models/laporan.md (dibuat ulang setiap latih.py dijalankan)."""
    m = b["metrik_test"]
    baris = [
        "# Ringkasan hasil - Kasus 1: Klasifikasi kultivar anggur", "",
        f"- Model terpilih: **{b['nama_model']}** {b['hyperparameter']}",
        f"- Varian fitur: {b['varian_fitur']}",
        f"- Split: {b['split']['n_train']} train / {b['split']['n_test']} test "
        f"(stratified, random_state={b['split']['random_state']})", "",
        "## Metrik data test (dievaluasi satu kali)", "", "| Metrik | Nilai |", "|---|---|"]
    for k in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
              "f1_weighted", "mcc", "log_loss", "roc_auc_ovr_macro",
              "baseline_akurasi_mayoritas"):
        baris.append(f"| {k} | {m[k]:.4f} |")
    lo, hi = m["accuracy_ci95"]
    baris += [f"| accuracy 95% CI (Wilson) | [{lo:.3f}, {hi:.3f}] |", "",
              "## Perbandingan algoritma (nested CV, data train)", "",
              "| Model | F1-macro (rata-rata) | Std | Accuracy |", "|---|---|---|---|"]
    for r in b["cv_model"]:
        baris.append(f"| {r['model']} | {r['f1_macro_mean']:.4f} | {r['f1_macro_std']:.4f} | "
                     f"{r['accuracy_mean']:.4f} |")
    baris += ["", "## Ablasi varian fitur (F1-macro CV, data train)", "",
              "| Varian | Rata-rata |", "|---|---|"]
    for r in b["ablasi_fitur"]:
        baris.append(f"| {r['varian']} | {r['rata_rata']:.4f} |")
    (DIR_MODEL / "laporan.md").write_text("\n".join(baris) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------
# Program utama
# ----------------------------------------------------------------------------------
def main() -> dict:
    DIR_MODEL.mkdir(exist_ok=True)

    # 1. Data dan audit ---------------------------------------------------------------
    mentah = F.muat_data()
    audit = F.audit_data(mentah)
    print(f"[audit] {audit['jumlah_baris']} baris | duplikat={audit['duplikat']} | "
          f"missing={audit['missing_total']} | nilai<=0={audit['nilai_tidak_wajar_<=0']}")
    print(f"[audit] kelas: {audit['distribusi_kelas']}")
    df = F.bersihkan(mentah)
    kelas = sorted(df[F.TARGET].unique())
    X, y = df[F.FITUR_ASLI], df[F.TARGET]

    # 2. SPLIT dulu, baru preprocessing apa pun ---------------------------------------
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                          random_state=RS)
    print(f"[split] train={len(Xtr)} test={len(Xte)} (stratified, random_state={RS})")

    # 3a. Ablasi varian fitur: CV di data train (model default, tanpa tuning)
    cv_abl = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
    keluarga = ["Logistic Regression", "SVM (RBF)", "Random Forest"]
    baris = []
    for varian, kw in F.VARIAN_FITUR.items():
        skor = {}
        for m in keluarga:
            r = cross_validate(F.buat_pipeline(m, **kw), Xtr, ytr, cv=cv_abl,
                               scoring="f1_macro", n_jobs=-1)
            skor[m] = float(r["test_score"].mean())
        baris.append({"varian": varian, **skor, "rata_rata": float(np.mean(list(skor.values())))})
    ablasi = pd.DataFrame(baris)
    print("\n[ablasi fitur] F1-macro CV (data train):")
    print(ablasi.round(4).to_string(index=False))
    # aturan keputusan: rata-rata F1-macro tertinggi; seri -> fitur lebih sedikit (urutan)
    terbaik = ablasi.sort_values("rata_rata", ascending=False, kind="stable").iloc[0]["varian"]
    kw_fitur = F.VARIAN_FITUR[terbaik]
    print(f"[ablasi fitur] varian terpilih: {terbaik}")

    # 3b. Nested CV: tuning di loop dalam, estimasi tak bias di loop luar -------------
    cv_luar = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RS)
    cv_dalam = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)
    skor_model = []
    print("\n[nested CV] membandingkan algoritma ...")
    for m in F.NAMA_MODEL:
        gs = GridSearchCV(F.buat_pipeline(m, **kw_fitur), F.GRID_TUNING[m], cv=cv_dalam,
                          scoring="f1_macro", n_jobs=1)
        r = cross_validate(gs, Xtr, ytr, cv=cv_luar, n_jobs=-1,
                           scoring={"f1_macro": "f1_macro", "accuracy": "accuracy",
                                    "balanced_accuracy": "balanced_accuracy"})
        skor_model.append({
            "model": m,
            "f1_macro_mean": float(r["test_f1_macro"].mean()),
            "f1_macro_std": float(r["test_f1_macro"].std()),
            "accuracy_mean": float(r["test_accuracy"].mean()),
            "balanced_accuracy_mean": float(r["test_balanced_accuracy"].mean()),
        })
        print(f"   {m:<20} F1-macro = {skor_model[-1]['f1_macro_mean']:.4f} "
              f"± {skor_model[-1]['f1_macro_std']:.4f}")
    hasil_cv = pd.DataFrame(skor_model)
    # pemilihan: F1-macro tertinggi; seri -> simpangan baku lebih kecil
    pilih = hasil_cv.sort_values(["f1_macro_mean", "f1_macro_std"],
                                 ascending=[False, True]).iloc[0]["model"]
    print(f"[pemilihan] model terpilih (berdasarkan CV di data train): {pilih}")

    # 4. Latih final di seluruh data train, evaluasi test SATU KALI ---------------------
    final_gs = GridSearchCV(F.buat_pipeline(pilih, **kw_fitur), F.GRID_TUNING[pilih],
                            cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                            scoring="f1_macro", n_jobs=-1, refit=True)
    final_gs.fit(Xtr, ytr)
    pipeline = final_gs.best_estimator_
    print(f"[final] hyperparameter terbaik: {final_gs.best_params_}")

    ypred = pipeline.predict(Xte)
    proba = pipeline.predict_proba(Xte)
    assert list(pipeline.classes_) == kelas
    metrik = hitung_metrik(yte, ypred, proba, kelas)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    metrik["baseline_akurasi_mayoritas"] = float(accuracy_score(yte, dummy.predict(Xte)))
    print("\n[TEST] (dievaluasi satu kali)")
    for k in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "log_loss",
              "roc_auc_ovr_macro"):
        print(f"   {k:<18} {metrik[k]:.4f}")
    print(f"   akurasi 95% CI     [{metrik['accuracy_ci95'][0]:.3f}, "
          f"{metrik['accuracy_ci95'][1]:.3f}] | salah {metrik['n_salah']}/{metrik['n_test']}")

    # Diagnostik (pada data train, bukan test): importansi permutasi fitur mentah
    pi = permutation_importance(pipeline, Xtr, ytr, scoring="f1_macro", n_repeats=20,
                                random_state=RS, n_jobs=-1)
    importansi = pd.DataFrame({"fitur": F.FITUR_ASLI, "rata_rata": pi.importances_mean,
                               "std": pi.importances_std})
    fitur_akhir = list(pipeline[:2].transform(Xtr.head(1)).columns)
    buat_gambar(Xtr, ytr, Xte, yte, ypred, kelas, hasil_cv, importansi,
                metrik["confusion_matrix"])

    # 5. Simpan pipeline + metadata ---------------------------------------------------
    rentang = {c: {"min": float(Xtr[c].min()), "max": float(Xtr[c].max()),
                   "median": float(Xtr[c].median()),
                   "q05": float(Xtr[c].quantile(0.05)), "q95": float(Xtr[c].quantile(0.95))}
               for c in F.FITUR_ASLI}
    median_kelas = {k: Xtr[ytr == k].median().to_dict() for k in kelas}
    bundle = {
        "pipeline": pipeline,
        "kasus": "Kasus 1 - Klasifikasi kultivar anggur",
        "nama_model": pilih,
        "hyperparameter": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                           for k, v in final_gs.best_params_.items()},
        "target": F.TARGET,
        "kelas": kelas,
        "fitur_mentah": F.FITUR_ASLI,
        "fitur_akhir": fitur_akhir,
        "varian_fitur": terbaik,
        "rentang_fitur": rentang,
        "median_kelas": median_kelas,
        "metrik_test": metrik,
        "cv_model": hasil_cv.to_dict("records"),
        "ablasi_fitur": ablasi.to_dict("records"),
        "importansi": importansi.sort_values("rata_rata", ascending=False).to_dict("records"),
        "audit": {k: v for k, v in audit.items() if k != "statistik"},
        "split": {"test_size": TEST_SIZE, "random_state": RS, "n_train": int(len(Xtr)),
                  "n_test": int(len(Xte))},
        "versi": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                  "pandas": pd.__version__, "numpy": np.__version__},
        "dilatih_pada": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, FILE_MODEL, compress=3)
    ringkas = {k: v for k, v in bundle.items() if k not in ("pipeline",)}
    (DIR_MODEL / "metrics.json").write_text(
        json.dumps(ringkas, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tulis_laporan_md(bundle)
    print(f"\n[simpan] {FILE_MODEL.relative_to(BASE_DIR)}, models/metrics.json, models/laporan.md")
    return bundle


if __name__ == "__main__":
    main()
