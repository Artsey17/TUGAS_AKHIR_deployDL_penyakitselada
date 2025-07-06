 # TUGAS AKHIR
"PENERAPAN DEEP LEARNING YOLO BERDASARKAN CITRA DIGITAL UNTUK MENDETEKSI PENYAKIT DAUN SELADA:  PERFORMA MODEL YOLOV11N DAN YOLOV12N BERBASIS RASPBERRY PI"

TUJUAN :

1.   Model manakah yang dapat paling efektif digunakan pada Penelitian Penyakit daun selada.
2.   Pemanfaatan Deep learning yaitu dengan arsitektur YOLO untuk analaisis prediksi bounding box penyakit daun selada.
3.  Penggunaan Optimizer yang memiliki performa terbaik pada dataset Penyakit tanaman selada.

METODE :
Arsitektur model YOLOv11 dan YOLOv12 dibandingkan hasil terbaik dilakukan fine tuning dengan Optimizer SGD dan AdamW pada dataset penyakit selada sebesar 50 epochs, Hasil optimizer terbaik dideploy Pada Raspberry Pi 3

Dataset :
1. Dataset Diambil dari sumber Kaggle, Data Real Lapangan, dan Roboflow. yang digabung menjad custom dataset.
2. Dataset split dataset menjadi 3 bagian : Training, Validation, dan Testing
  > Dibagi menjadi 3 yaitu :
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
>Data akan dilakukan 2 kali pelatihan pada awalnya akan dilatih dengan model YOLOv11/v12 yang telah ditraining di COCO kemudian model dilakukan training dengan yang bervariasi hingga mendapatkan model g, dan dilakukan analisis model mana yang terbaik.

> a. YOLOv11nano dan YOLOv12n 1 kali epochs (50)  dataset 18 (70.20.10):


> b. YOLOv11nano dan YOLOv12n 1 kali epochs (50)  dataset 12 (80.10.10):


> Hasil dari model tersebut akan dilakukan Analisis lebih lanjut yaitu dengan parameter
 1. mAP 50
 2. mAP 50-95
 3. F1 Confidence Score
 4. Speed Inference
 5. Size model
 6. Precision
 7. Recall
 8. Confusion Matrix
 9. Loss

Dari perbandingan hasil dari kedua model YOLO ini diambil dengan keseimbangan kecepatan serta akurasi yang mumpuni didapatkan YOLov11n secara keseluruhan mumpuni karena seimbang antara akurasi dan kecepatan. setelah didapatkan model yang sesuai dari eksperimen maka akan dilakukan analisis lanjutan.
langkah lanjutan ini adalah dari hasil model terbaik yaitu Model YOLOv11n ini akan dilakukan pergantian Optimizer 5x  Dataset 80/10/10 yang akan diganti dengna optimizer AdamW dan juga SGD. sehingga didapatkan.

Data hasil Optimizer dengan dataset 80/10/10
Optimizer yang digunakan adalah SGD dan AdamW yang dibagi menjadi learning rate
>  SGD : [0.02, 0.01, 0.005, 0.002, 0.001]
AdamW : [0.002, 0.001, 0.0005, 0.0002, 0.0001]

data hasil optimizer : YOLOv11n dengan Optimizer SGD pada_ learning rate_ 0.005.
Model yang terbaik akan dipakai pada Raspberry Pi 3 untuk menganalisis penyakit tanaman selada.
kemudian saya buat alat dengan tampilan Waveshare lcd 7 inci yang diintegrasikan dengan rapsberry pi 3 dan menggunakna GUI tkinter untuk membuat tampilan gui, pada menu terdapat upload gambar dan upload video serta video real time dapat menggunakan raspi cam atau webcam.
