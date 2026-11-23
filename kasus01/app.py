"""
app.py - Aplikasi Streamlit Kasus 1: Prediksi kultivar anggur dari analisis kimia.

Jalankan:  streamlit run app.py
Aplikasi menerima input MENTAH (13 hasil analisis kimia), memanggil pipeline yang sudah
disimpan dengan joblib (imputasi -> fitur rasio -> scaling -> model), lalu menampilkan
prediksi + probabilitas, dan metrik evaluasi hasil pelatihan (data test, satu kali).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))  # agar pickle dapat menemukan kelas FiturRasio di fitur.py

import fitur as F  # noqa: E402

FILE_MODEL = BASE_DIR / "models" / "wine_kultivar.joblib"
DIR_FIGUR = BASE_DIR / "models" / "figures"

st.set_page_config(page_title="Klasifikasi Kultivar Anggur", page_icon="🍷", layout="wide")

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(120deg,#5b1a2b 0%,#8e2c48 60%,#c44536 100%);
           padding: 1.4rem 1.8rem; border-radius: 14px; color: #fff; margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
    .hero p {margin: .3rem 0 0 0; opacity: .92;}
    .hasil {border-radius: 14px; padding: 1.2rem 1.4rem; background: rgba(142,44,72,.10);
            border: 1px solid rgba(142,44,72,.35);}
    .hasil h2 {margin: 0; font-size: 2rem;}
    </style>
    <div class="hero">
      <h1>🍷 Klasifikasi Kultivar Anggur</h1>
      <p>Menentukan kultivar (A, B, C) sebuah sampel anggur dari 13 hasil analisis kimia.
         UTS Intelegensi Buatan · Kasus 1</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------------
# Memuat model (dilatih otomatis bila file joblib belum ada)
# ----------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model ...")
def muat_model():
    try:
        return joblib.load(FILE_MODEL)
    except Exception:  # file belum ada atau tidak cocok dengan versi library
        import latih  # melatih ulang dan menyimpan ke models/

        latih.main()
        return joblib.load(FILE_MODEL)


bundle = muat_model()
pipe = bundle["pipeline"]
kelas = bundle["kelas"]
rentang = bundle["rentang_fitur"]
metrik = bundle["metrik_test"]

# ----------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.header("Ringkasan model")
    st.write(f"**Algoritma:** {bundle['nama_model']}")
    st.write(f"**Varian fitur:** {bundle['varian_fitur']}")
    st.write(f"**Data:** {bundle['split']['n_train']} train · {bundle['split']['n_test']} test")
    st.metric("Akurasi (test)", f"{metrik['accuracy']:.1%}")
    st.metric("F1-macro (test)", f"{metrik['f1_macro']:.3f}")
    st.caption(f"Dilatih: {bundle['dilatih_pada']} · scikit-learn {bundle['versi']['sklearn']}")
    st.divider()
    st.caption("Kelas: " + ", ".join(kelas))

tab_pred, tab_metrik, tab_batch, tab_tentang = st.tabs(
    ["🔮 Prediksi", "📊 Metrik evaluasi", "📁 Prediksi batch (CSV)", "ℹ️ Tentang"])

# ----------------------------------------------------------------------------------
# Tab 1: prediksi satu sampel
# ----------------------------------------------------------------------------------
KELOMPOK = {
    "Komposisi dasar": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium"],
    "Komponen fenolik": ["total_phenols", "flavanoids", "nonflavanoid_phenols",
                         "proanthocyanins"],
    "Warna dan lainnya": ["color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"],
}
PRESET = ["Median seluruh data train"] + [f"Contoh {k}" for k in kelas] + ["Isi manual"]


def terapkan_preset():
    pilihan = st.session_state["preset"]
    if pilihan == "Isi manual":
        return
    if pilihan.startswith("Contoh "):
        sumber = bundle["median_kelas"][pilihan.replace("Contoh ", "")]
    else:
        sumber = {f: rentang[f]["median"] for f in F.FITUR_ASLI}
    for f in F.FITUR_ASLI:
        st.session_state[f"in_{f}"] = float(sumber[f])


for f in F.FITUR_ASLI:
    st.session_state.setdefault(f"in_{f}", float(rentang[f]["median"]))

with tab_pred:
    st.subheader("Masukkan hasil analisis kimia")
    st.selectbox("Isi cepat dengan nilai contoh", PRESET, key="preset",
                 on_change=terapkan_preset,
                 help="Nilai contoh diambil dari median tiap kelas pada data train.")
    kolom = st.columns(3)
    for kol, (judul, daftar) in zip(kolom, KELOMPOK.items()):
        with kol:
            st.markdown(f"**{judul}**")
            for f in daftar:
                label, satuan, ket, step, fmt = F.INFO_FITUR[f]
                st.number_input(f"{label} {f'({satuan})' if satuan else ''}".strip(),
                                min_value=0.0, max_value=float(rentang[f]["max"]) * 3,
                                step=float(step), format=fmt, key=f"in_{f}", help=ket)

    masukan = pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in F.FITUR_ASLI}])
    luar = [F.INFO_FITUR[f][0] for f in F.FITUR_ASLI
            if not rentang[f]["min"] <= masukan.at[0, f] <= rentang[f]["max"]]
    if luar:
        st.warning("Nilai di luar rentang data latih (model melakukan ekstrapolasi, "
                   "prediksi kurang andal): " + ", ".join(luar))
    if (masukan.iloc[0] <= 0).any():
        st.info("Nilai 0 dianggap tidak tercatat dan diimputasi dengan median data train.")
    masukan_bersih = masukan.where(masukan > 0)  # 0 -> NaN, diimputasi oleh pipeline

    if st.button("Prediksi kultivar", type="primary"):
        proba = pipe.predict_proba(masukan_bersih)[0]
        hasil = pd.DataFrame({"Kultivar": pipe.classes_, "Probabilitas": proba}) \
            .sort_values("Probabilitas", ascending=False).reset_index(drop=True)
        pred, yakin = hasil.at[0, "Kultivar"], hasil.at[0, "Probabilitas"]

        k1, k2 = st.columns([1, 1.4])
        with k1:
            st.markdown(f'<div class="hasil"><span>Prediksi</span><h2>{pred}</h2>'
                        f'<span>Keyakinan model: <b>{yakin:.1%}</b></span></div>',
                        unsafe_allow_html=True)
            if yakin < 0.6:
                st.warning("Keyakinan rendah: sampel berada dekat batas antar kultivar. "
                           "Periksa kembali input atau lakukan analisis lanjutan.")
            elif yakin < 0.85:
                st.info("Keyakinan sedang.")
            else:
                st.success("Keyakinan tinggi.")
        with k2:
            st.markdown("**Probabilitas per kultivar**")
            st.bar_chart(hasil.set_index("Kultivar")["Probabilitas"])
        tampil = hasil.copy()
        tampil["Probabilitas"] = tampil["Probabilitas"].map("{:.2%}".format)
        st.dataframe(tampil, hide_index=True)

        with st.expander("Lihat fitur setelah feature engineering (input pipeline ke model)"):
            diolah = pipe[:2].transform(masukan_bersih)
            st.dataframe(diolah.T.rename(columns={0: "nilai"}))
            st.caption("Fitur rasio dihitung otomatis dari input mentah; pengguna tidak perlu "
                       "memasukkannya.")

# ----------------------------------------------------------------------------------
# Tab 2: metrik evaluasi
# ----------------------------------------------------------------------------------
with tab_metrik:
    st.subheader("Hasil evaluasi pada data test (dievaluasi satu kali)")
    c = st.columns(5)
    c[0].metric("Accuracy", f"{metrik['accuracy']:.3f}")
    c[1].metric("Balanced accuracy", f"{metrik['balanced_accuracy']:.3f}")
    c[2].metric("F1-macro", f"{metrik['f1_macro']:.3f}")
    c[3].metric("MCC", f"{metrik['mcc']:.3f}")
    c[4].metric("Log loss", f"{metrik['log_loss']:.3f}")
    lo, hi = metrik["accuracy_ci95"]
    st.caption(f"Akurasi 95% CI (Wilson): [{lo:.3f}, {hi:.3f}] · baseline kelas mayoritas: "
               f"{metrik['baseline_akurasi_mayoritas']:.3f} · ROC-AUC OvR macro: "
               f"{metrik['roc_auc_ovr_macro']:.3f} · salah {metrik['n_salah']} dari "
               f"{metrik['n_test']} sampel test")

    kiri, kanan = st.columns(2)
    with kiri:
        st.markdown("**Confusion matrix** (baris = aktual, kolom = prediksi)")
        st.dataframe(pd.DataFrame(metrik["confusion_matrix"], index=[f"Aktual {k}" for k in kelas],
                                  columns=[f"Pred {k}" for k in kelas]))
        if (DIR_FIGUR / "confusion_matrix.png").exists():
            st.image(str(DIR_FIGUR / "confusion_matrix.png"))
    with kanan:
        st.markdown("**Laporan per kelas**")
        lap = pd.DataFrame(metrik["laporan_kelas"]).T.loc[kelas + ["macro avg", "weighted avg"]]
        lap["support"] = lap["support"].astype(int)
        st.dataframe(lap.round(3))
        st.markdown("**Perbandingan algoritma** (nested CV di data train)")
        cv = pd.DataFrame(bundle["cv_model"]).set_index("model")
        st.dataframe(cv.rename(columns={
            "f1_macro_mean": "F1-macro (rata-rata)", "f1_macro_std": "F1-macro (std)",
            "accuracy_mean": "Accuracy", "balanced_accuracy_mean": "Balanced acc."}).round(4))

    st.markdown("**Ablasi varian fitur** (F1-macro CV 5×3, data train)")
    st.dataframe(pd.DataFrame(bundle["ablasi_fitur"]).set_index("varian").round(4))

    g1, g2 = st.columns(2)
    for kol, nama in ((g1, "perbandingan_model.png"), (g2, "importansi.png")):
        if (DIR_FIGUR / nama).exists():
            kol.image(str(DIR_FIGUR / nama))

    with st.expander("Apa makna metrik-metrik ini?"):
        st.markdown(
            """
- **Accuracy**: proporsi sampel yang diprediksi benar. Cukup informatif di sini karena
  kelas relatif seimbang (A 33%, B 40%, C 27%).
- **Balanced accuracy**: rata-rata recall tiap kelas; tidak terpengaruh ukuran kelas.
- **Precision / Recall / F1 (macro)**: dihitung per kultivar lalu dirata-ratakan sehingga
  tiap kultivar sama penting. Precision = dari yang diprediksi kultivar X, berapa yang
  benar; recall = dari kultivar X yang sebenarnya, berapa yang terdeteksi.
- **MCC**: korelasi prediksi–aktual (−1 s.d. 1); ketat untuk multikelas.
- **Log loss**: menilai kualitas probabilitas, bukan hanya label; makin kecil makin baik.
- **ROC-AUC OvR**: kemampuan memisahkan satu kultivar dari yang lain di semua ambang.
- **Confusion matrix**: memperlihatkan kultivar mana yang tertukar dengan mana.
            """)

# ----------------------------------------------------------------------------------
# Tab 3: prediksi batch
# ----------------------------------------------------------------------------------
with tab_batch:
    st.subheader("Prediksi banyak sampel sekaligus")
    st.write("Unggah CSV berisi 13 kolom analisis kimia (nama kolom sama seperti dataset; "
             "kolom `kultivar` bila ada akan diabaikan).")
    contoh = pd.DataFrame([{f: rentang[f]["median"] for f in F.FITUR_ASLI}]).round(3)
    st.download_button("Unduh templat CSV", contoh.to_csv(index=False).encode("utf-8"),
                       file_name="templat_wine.csv", mime="text/csv")
    berkas = st.file_uploader("Berkas CSV", type="csv")
    if berkas is not None:
        data = pd.read_csv(berkas)
        data.columns = [F.normalisasi_kolom(k) for k in data.columns]
        kurang = [f for f in F.FITUR_ASLI if f not in data.columns]
        if kurang:
            st.error("Kolom tidak ditemukan: " + ", ".join(kurang))
        else:
            x = data[F.FITUR_ASLI].apply(pd.to_numeric, errors="coerce")
            x = x.where(x > 0)
            pb = pipe.predict_proba(x)
            keluar = data.copy()
            keluar["prediksi_kultivar"] = pipe.classes_[pb.argmax(axis=1)]
            for i, k in enumerate(pipe.classes_):
                keluar[f"prob_{k}"] = pb[:, i].round(4)
            st.dataframe(keluar)
            st.download_button("Unduh hasil prediksi", keluar.to_csv(index=False).encode("utf-8"),
                               file_name="hasil_prediksi.csv", mime="text/csv")

# ----------------------------------------------------------------------------------
# Tab 4: tentang
# ----------------------------------------------------------------------------------
with tab_tentang:
    st.subheader("Keputusan desain")
    st.markdown(
        f"""
- **Split sebelum preprocessing**: data dibagi 80/20 (stratified, `random_state=42`) sebelum
  imputasi, scaling, atau fitur apa pun. Median imputasi dan mean/std scaler hanya dari train.
- **Fitur**: 13 hasil analisis kimia dipakai semua. Tidak ada kolom ID atau kolom yang baru
  diketahui setelah kejadian, jadi tidak ada kebocoran target.
- **Fitur rasio** (6): {", ".join(f"`{k}`" for k in F.RASIO)}.
- **Varian terpilih**: {bundle['varian_fitur']} (rata-rata F1-macro CV tertinggi di train).
  `total_phenols` dan `flavanoids` dibuang karena korelasinya 0,86 dan informasinya tercakup
  oleh rasio.
- **Scaling**: `StandardScaler` di dalam pipeline karena skala fitur sangat berbeda
  (prolin ratusan–ribuan, hue 0,5–1,7).
- **Pemilihan model**: nested cross-validation di data train (tuning di loop dalam);
  data test hanya dipakai sekali pada akhir.

**Keterbatasan.** Dataset kecil (178 sampel, test 36 sampel) dan tergolong mudah, sehingga
akurasi test sangat tinggi tetapi selang kepercayaannya lebar. Model hanya berlaku untuk
anggur dari tiga kultivar di wilayah yang sama dan tidak untuk kultivar lain.
        """)
