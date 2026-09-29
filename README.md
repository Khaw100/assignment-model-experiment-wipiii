# AI Model Experiment & Evaluation — Sentiment Analysis

Eksperimen perbandingan dua pendekatan AI untuk **sentiment analysis ulasan pelanggan e-commerce**:

1. **Model Klasik (Scikit-learn)** — TF-IDF + Logistic Regression, dilatih sendiri.
2. **LLM API (Gemini)** — klasifikasi via prompting (zero-shot), tanpa training.

Keduanya dievaluasi pada **test set yang sama** dengan metrik Accuracy, Precision, Recall, F1-Score + confusion matrix, lalu dibandingkan untuk menyusun rekomendasi technical approach.

---

## 1. Problem Statement

### Objective
Tim produk e-commerce ingin fitur otomatis yang mengklasifikasikan sentimen ulasan pelanggan (positif/negatif) pada halaman produk, sebelum memutuskan pendekatan untuk produksi. Eksperimen ini membandingkan dua kandidat dan memberi rekomendasi berbasis data.

### Target/Label
- **Input**: teks ulasan (`review_text`)
- **Output**: `positif` / `negatif` (kolom `sentiment`)

### Batasan & Asumsi
- Klasifikasi biner; ulasan diasumsikan polar (tanpa kelas netral).
- Test set identik untuk kedua pendekatan (split `random_state=42`, stratified).
- Preprocessing klasik dibuat sesederhana mungkin (TF-IDF murni, tanpa stemming/stopword bahasa Indonesia) sebagai baseline.
- Penggunaan Gemini melalui `google-genai` SDK, key via environment variable `GEMINI_API_KEY`.
- Biaya API belum dihitung dari tagihan riil, hanya estimasi volume token.

## 2. Dataset

`data/customer_reviews_sentiment.csv` — 200 baris ulasan produk e-commerce (kolom: `review_id`, `product_name`, `review_text`, `sentiment`), 110 positif / 90 negatif.

> Catatan: file asli tidak diubah; salinan test set (`data/test_set.csv`) hanya output evaluasi, bukan pengganti dataset.

## 3. Metodologi

1. **Split**: 80/20 stratified (train 160, test 40), `random_state=42`. Test set yang sama dipakai untuk kedua pendekatan.
2. **Model klasik**: `TfidfVectorizer` → `LogisticRegression(max_iter=1000)`, prediksi pada test set.
3. **LLM (Gemini)**: prompt zero-shot per ulasan (`gemini-2.0-flash`, `temperature=0.2`, `max_output_tokens=8`), output diparse ke `positif`/`negatif` (jawaban di luar dua label dicatat `invalid`).
4. **Evaluasi**: Accuracy, Precision, Recall, F1 (positif = kelas positif), confusion matrix, tabel perbandingan.

### Hasil (jalankan notebook dengan `GEMINI_API_KEY` untuk melengkapi kolom LLM)

Model klasik pada test set: **Accuracy 1.00, Precision 1.00, Recall 1.00, F1 1.00** (CM: 18 TN, 0 FP, 0 FN, 22 TP).

> ⚠️ **Peringatan interpretasi**: dataset ini hanya memuat 40 teks unik dari 200 baris (banyak duplikat) sehingga test set mengandung baris identik dengan train (data leakage dari sisi data, bukan proses). Metrik 1.0 mencerminkan kemudahan task + duplikasi, bukan performa produksi. Validasi silang pada 40 teks unik saja menghasilkan rata-rata akurasi ~0.7-0.85 tergantung fold — lebih realistis.

## 4. Tabel Perbandingan

| Pendekatan | Accuracy | Precision (positif) | Recall (positif) | F1 (positif) |
|---|---|---|---|---|
| Model Klasik (TF-IDF + LogReg) | 1.00 | 1.00 | 1.00 | 1.00 |
| LLM API (Gemini) | _isi setelah run dengan API key_ | | | |

## 5. Analisis Trade-off dan Limitation

### Performa
- Di task polar sederhana ini keduanya diperkirakan sangat akurat; model klasik mencapai 1.0 pada test set (dengan caveat leakage di atas).
- Untuk ulasan baru dengan gaya bahasa yang belum pernah dilihat, model klasik terikat vocabulary TF-IDF; LLM lebih tahan ke variasi bahasa.

### Effort / Kecepatan / Biaya

| Aspek | Model Klasik | LLM API (Gemini) |
|---|---|---|
| Effort implementasi | Perlu data berlabel, pipeline, training, evaluasi (sekali setup) | Prompt + API call; bisa produksi dalam hitungan jam |
| Kecepatan per prediksi | Milidetik (lokal) | Ratusan ms - detik (network) |
| Biaya | Compute sendiri; tanpa biaya per-request | Berbasis token; membesar dengan volume; ada rate-limit |
| Data privacy | Data tetap internal | Setiap ulasan dikirim ke pihak ketiga |
| Interpretabilitas | Koefisien bisa diinspeksi | Black-box |
| Maintenance | Retrain saat distribusi data bergeser | Penyesuaian prompt; risiko perubahan perilaku model provider |

### Limitation
- Dataset kecil dan sangat berduplikat -> metrik terlihat terlalu bagus; kesimpulan produksi harus pakai data baru.
- Label biner menghilangkan kasus netral/campuran.
- LLM non-deterministik meski temperature rendah; hasil antar-run bisa sedikit beda.
- Model klasik gagal pada kata di luar vocabulary; LLM bisa salah kalau ulasan menyampur pujian + komplain tanpa konteks fitur.

## 6. Rekomendasi Technical Approach

**MVP: pakai Gemini API (LLM) terlebih dahulu; pertahankan model klasik sebagai baseline dan rencana migrasi.**

Alasan:
1. Kualitas prediksi pada task polar ini setara/sangat baik tanpa biaya training; time-to-market tercepat.
2. Robust terhadap kosakata baru tanpa retraining.
3. Biaya per ulasan rendah pada skala awal; ulasan pendek berarti prompt murah.

Migrasi ke model klasik (atau hybrid LLM + model klasik) bila: volume harian besar (biaya API melebihi biaya maintain pipeline), kebutuhan latensi sangat ketat, atau regulasi melarang data keluar. Dengan label hasil LLM yang terkumpul, model klasik bisa dilatih ulang berkala (distant supervision).

## 7. Cara Menjalankan

```bash
# 1) siapkan environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) set API key Gemini (untuk bagian LLM; bagian klasik jalan tanpa ini)
export GEMINI_API_KEY='<key kamu>'    # Windows CMD: set GEMINI_API_KEY=... / PowerShell: $env:GEMINI_API_KEY="..."

# 3) jalankan notebook
cd notebook
jupyter notebook experiment_notebook.ipynb   # lalu Run All
# atau non-interaktif:
jupyter nbconvert --to notebook --execute --inplace experiment_notebook.ipynb
```

Output penting: tabel perbandingan (Section 6.3) dan grafik confusion matrix tersimpan ke `documentation/model_comparison_summary.png`.

## 8. Struktur Repo

```
model-experiment-assignment/
├── data/
│   ├── customer_reviews_sentiment.csv   # dataset asli (tidak diubah)
│   └── test_set.csv                     # test set yang dipakai evaluasi
├── notebook/
│   └── experiment_notebook.ipynb        # eksperimen kedua pendekatan
├── documentation/
│   └── model_comparison_summary.png     # visual confusion matrix
├── README.md
└── requirements.txt
```
