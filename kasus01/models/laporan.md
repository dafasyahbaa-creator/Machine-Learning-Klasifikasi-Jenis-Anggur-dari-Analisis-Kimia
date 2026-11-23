# Ringkasan hasil - Kasus 1: Klasifikasi kultivar anggur

- Model terpilih: **SVM (RBF)** {'model__C': 1.0, 'model__gamma': 'scale'}
- Varian fitur: C. Rasio, tanpa total_phenols & flavanoids (17)
- Split: 142 train / 36 test (stratified, random_state=42)

## Metrik data test (dievaluasi satu kali)

| Metrik | Nilai |
|---|---|
| accuracy | 1.0000 |
| balanced_accuracy | 1.0000 |
| precision_macro | 1.0000 |
| recall_macro | 1.0000 |
| f1_macro | 1.0000 |
| f1_weighted | 1.0000 |
| mcc | 1.0000 |
| log_loss | 0.0586 |
| roc_auc_ovr_macro | 1.0000 |
| baseline_akurasi_mayoritas | 0.3889 |
| accuracy 95% CI (Wilson) | [0.904, 1.000] |

## Perbandingan algoritma (nested CV, data train)

| Model | F1-macro (rata-rata) | Std | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.9859 | 0.0230 | 0.9861 |
| SVM (RBF) | 1.0000 | 0.0000 | 1.0000 |
| KNN | 0.9667 | 0.0220 | 0.9648 |
| Random Forest | 0.9791 | 0.0283 | 0.9789 |
| Gradient Boosting | 0.9557 | 0.0501 | 0.9546 |

## Ablasi varian fitur (F1-macro CV, data train)

| Varian | Rata-rata |
|---|---|
| A. Fitur asli saja (13) | 0.9838 |
| B. Asli + rasio (19) | 0.9865 |
| C. Rasio, tanpa total_phenols & flavanoids (17) | 0.9887 |
