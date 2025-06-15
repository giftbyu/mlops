Berdasarkan analisis perbandingan yang telah dilakukan, dapat disimpulkan bahwa pipeline preprocessing data yang diotomatisasi melalui skrip Python (preprocess.py) telah berhasil divalidasi dan terbukti konsisten secara fungsional dengan proses eksperimen manual yang dilakukan di Jupyter Notebook. Artefak data yang dihasilkan oleh skrip otomatis kini dapat dianggap sebagai "sumber kebenaran tunggal" (single source of truth) yang andal untuk tahap pemodelan selanjutnya.

# **Temuan Utama Validasi**:

Proses validasi dilakukan dengan membandingkan output dari kedua alur kerja:

Konsistensi Struktural: Telah dipastikan bahwa kedua file output (.csv) memiliki struktur yang identik, termasuk jumlah baris, jumlah kolom, nama kolom, dan tipe data di setiap kolom. Isu awal terkait perbedaan struktur telah berhasil diidentifikasi dan diperbaiki.

# **Konsistensi Fungsional**:

Perbandingan konten menggunakan metode kesamaan absolut (df.equals()) menunjukkan adanya perbedaan.

Namun, analisis lebih lanjut menggunakan perbandingan numerik dengan toleransi (np.allclose()) mengonfirmasi bahwa perbedaan tersebut sangat kecil (di bawah 1e-5).

Perbedaan minor ini merupakan artefak komputasi yang wajar dan dapat diatribusikan pada variasi floating-point arithmetic antara lingkungan eksekusi yang berbeda. Perbedaan ini tidak signifikan secara statistik dan tidak akan memengaruhi performa model.

**Reproduktifitas**: Tujuan utama untuk menciptakan sebuah proses yang dapat direproduksi telah tercapai. Skrip otomatis (preprocess.py) secara konsisten menerapkan logika yang sama (imputasi, penanganan outlier, feature engineering, dan normalisasi) seperti yang telah dieksplorasi dan divalidasi dalam notebook.


![image](https://github.com/user-attachments/assets/e1b9939c-7fae-4fd0-bc74-5ab674eb328a)
