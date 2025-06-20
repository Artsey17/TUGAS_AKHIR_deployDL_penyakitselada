# TUGAS_AKHIR_deployDL_penyakitselada
 # TUGAS AKHIR
"Studi Komparatif Kinerja Optimizer AdamW dan SGD untuk Peningkatan Akurasi dan Kecepatan Deteksi Penyakit Daun Selada Menggunakan Model YOLOv11n Berbasis Raspberry Pi"

TUJUAN :

1. Menentukan arsitektur model dan konfigurasi pembagian data yang paling optimal sebagai baseline untuk deteksi penyakit daun selada.
2. Melakukan studi komparatif mendalam untuk menemukan konfigurasi optimizer dan learning rate yang paling unggul (AdamW vs. SGD) guna memaksimalkan performa model baseline.
3. Menghasilkan sebuah model final yang tervalidasi memiliki akurasi dan kecepatan yang mumpuni untuk kebutuhan implementasi pada platform Raspberry Pi.

METODE :
Menggunakan arsitektur model YOLOv11 dan YOLOv12 yang akan di fine tuning dengan 2 optimizer : SGD dan AdamW pada dataset sebanyak 50 epochs dan digunakan di embedded sistem pada raspberry pi 3.

Dataset :
1. Dataset Diambil dari sumber Kaggle, Data Real Lapangan, dan Roboflow. yang digabung menjad custom dataset.
2. Dataset split dataset menjadi 3 bagian : Training, Validation, dan Testing
  > Dibagi menjadi 2 yaitu :
  1. 80/10/10 = berisi 18102 gambar
  2. 70/20/10 = berisi 15906 gambar
3. Dataset memiliki 6 kelas yaitu :
  1. bacteria = Untuk kelas daun tanaman yang terkena bakteri (kuning bercak)
  2. Downy_mildew_on_lettuce : jamur pada daun membuat menguning (kuning bercak)
  3. Healthy = Daun selada yang sehat (Hijau)
  4. Lettuce Mosaic Virus = Virus yang menjangkit selada (kuning dan bercak hitam)
  5. Powdery_mildew_on_lettuce = Penyaakit jamur (putih)
  6. Septoria_Blight_on_lettuce = penyakit yang disebabkan oleh septoria sejenis jamur (hitam dan menguning bercak)

Pembahasan :
> Data akan dilakukan 2 kali pelatihan pada awalnya akan dilatih dengan model YOLOv11/v12 yang telah ditraining di COCO kemudian model dilakukan training dengan yang bervariasi hingga mendapatkan model setelah itu di re training atau continue learning, dan dilakukan analisis model mana yang terbaik.
> dari model yang diambil YOLO yang terbaik seimbang antara kecepatan dan akurasinya akan dipilih berdasarkan hasil kinerja parameter dari model
> model yang terbaik akan digunakan untuk diujicoba kembali dengan optimizer SGD dan AdamW dengan eksplorasi eksperimen dengan learning Rate
> Model yang terbaik akan dipakai pada Raspberry Pi 3 untuk menganalisis penyakit tanaman selada.

> YOLOv11nano :
  3. 80/10/10 + 80/10/10 =
  4. 70/20/10 + 70/20/10 =

> YOLOv12nano :
  3. 80/10/10 + 80/10/10 =
  4. 70/20/10 + 70/20/10 =

> Hasil dari model tersebut akan dilakukan Analisis lebih lanjut yaitu dengan parameter
 1. mAP 50
 2. mAP 50-95
 3. F1 Confidence Score
 4. Speed Inference
 5. Size model
 5. Box loss, val loss
 6. Precision
 7. Recall
 8. FPS pada Raspberry
 9. Confusion Matrix
 10. Metrics Precision
> Learning rate yang digunakna Di Optimizer SGD adalah : (0.02, 0.01, 0.005, 0.002, 0.001)
> Learning rate yang digunakna Di Optimizer AdamW adalah : (0.002, 0.001, 0.0005, 0.0002, 0.0001)
> Model yang paling baik Akan diimplemnentasikan di raspberry



