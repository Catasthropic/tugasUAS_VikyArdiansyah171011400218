# Tugas UAS Image Processing Viky Ardiansyah 171011400218 (07TPLE004)

Repositori ini dibuat untuk memenuhi tugas UAS Viky Ardiansyah (171011400218), Menggunakan modul pyimage untuk memproses gambar dengan bangun sederhana. Beberapa bangun yang dapat terdeteksi oleh image prosesing ini seputar bangun :
1. Persegi
2. Persegi panjang horizontal
3. Persegi panjang vertikal
4. Diamond
5. Segi lima
6. Segi delapan
7. Lingkaran


## Cara Kerja

Solusi ini mengadopsi dua teknik dasar untuk memperoleh hasil akhir dengan perluasan operasi pemrosesan gambar lainnya.
1. Deteksi Tepi Canny
2. Segmentasi Berbasis Warna
3. Prediksi Bounding Box

Pertama, ia menjalankan Canny Edge Detection untuk mendeteksi semua tepi tajam dalam sebuah gambar. Kemudian ia mengekstrak tepi yang tertutup saja dan menghilangkan semua yang lain termasuk garis dan kurva (bentuk rambu jalan selalu berupa gambar tertutup). Setelah itu, mempertimbangkan bentuk tertutup yang memiliki luas lebih dari 30% dari keseluruhan area gambar (tanda di gambar yang dipotong berisi bagian gambar yang lebih besar). Filter ini menghapus semua bentuk kecil yang tidak diperlukan termasuk teks di dalam tanda. Kemudian menghitung rasio r keliling suatu bentuk dengan luasnya dan mempertimbangkan rasio yang terkecil. Teknik ini menghilangkan semua bentuk tidak beraturan seperti yang ditunjukkan di bawah ini:
<img src="https://github.com/NaumanHSA/road-signs-shapes-detection-opencv/blob/main/ScreenShots/_ratio.png" width=1000/>


Setelah bentuk yang diinginkan diekstraksi, akhirnya kami menerapkan beberapa metode pendekatan kontur, termasuk Convex Hull dan approxPolyDP untuk terlebih dahulu mengisi setiap bagian yang rusak dari bentuk dan kemudian menemukan jumlah simpul yang memberi kita bentuk akhir.

Segmentasi Berbasis Warna digunakan saat Deteksi Tepi Canny gagal mengekstrak kontur yang diinginkan (dalam beberapa kasus, gambar diperbesar dan batas tanda dipotong). Dalam hal ini, kita menggunakan segmentasi berbasis warna yang mengekstrak warna dominan pada gambar (mungkin warna tanda jalan). Kita kemudian menghitung kontur dan mengulangi langkah-langkah di atas.

Ketiga, bahkan jika segmentasi berbasis warna gagal berfungsi, kita coba menghitung prediksi kotak pembatas pada semua kontur (dalam area kecil) dan menghitung kotak di sekitarnya. Kotak ini kemudian dicentang apakah bentuknya persegi, persegi panjang atau berlian.

## Requirements
    
    python: 3.x
    matplotlib: 3.2.0
    numpy: 1.18.5
    scikit_learn: 0.23.2
    opencv-python: 4.2.0
    

## Run code
Clone repositori dan buka direktori root. Masukkan perintah berikut dengan dua bendera untuk ditentukan.    
    
    python run.py --images_path=images --verbose=1
    
### Flags:
1. **--images_path** (default ./images) : menentukan jalur ke direktori gambar. Direktori harus berisi gambar.
2. **--verbose** (default 1): menentukan tingkat verbositas. Salah satu dari 1 atau 0. Mencetak bentuk di terminal saat verbose = 0 lain memvisualisasikan langkah-langkah yang terlibat saat memproses gambar secara grafis.

 
 
 # Referensi
 
 Kuantisasi Warna (KMeans Clustering)
 https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
 
 Deteksi bentuk dengan OpenCV
 https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
 
 Fitur Kontur
 https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
 
 Segmentasi Berbasis Warna
 https://realpython.com/python-opencv-color-spaces/
