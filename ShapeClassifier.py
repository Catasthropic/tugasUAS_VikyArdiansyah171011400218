"""
Tugas UAS Viky Ardiansyah
171011400218
07TPLE004

Image Processing menggunakan OpenCV untuk mendeteksi objek sederhana
IDE: Pycharm

"""

import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import cv2
import numpy as np


class ShapeClassifier:

    # Inisiasi Variabel
    def __init__(self, verbose=1):
        self.verbose = verbose
        self.image_path = None
        # batas warna yang digunakan untuk melakukan segmentasi berbasis warna
        self.colors = {
            "low_red": np.array([0, 193, 110]),
            "high_red": np.array([185, 255, 255]),
            "low_blue": np.array([110, 50, 50]),
            "high_blue": np.array([130, 255, 255]),
            "low_green": np.array([25, 80, 80]),
            "high_green": np.array([85, 255, 255]),
            "low_yellow": np.array([14, 85, 124]),
            "high_yellow": np.array([45, 255, 255]),
        }

    def _countors_based_segmentation(self, _image, contours, flag):
        # menghitung area gambar
        _area_image = _image.shape[0] * _image.shape[1]
        # pisahkan kontur dengan luas lebih dari 30% dari area gambar
        _large_area_contours = [cnt for cnt in contours if cv2.contourArea(cnt) / _area_image > 0.3]

        # jika tidak ada kontur yang memiliki luas lebih dari 30% dari luas gambar
        if len(_large_area_contours) == 0:
            if flag == 0:
                # melakukan segmentasi berbasis warna
                self.color_based_segmentation(_image)
            else:
                # melakukan prediksi kotak menggunakan kontur kecil
                box_params = self.box_prediction(contours)
                if self.verbose == 1:
                    # menampilkan hasil
                    self.display_box_results(_image, contours, box_params)
                else:
                    if len(box_params) > 0:
                        print("{}: {}".format(self.image_path, box_params[4]))
            return

        # ekstrak kontur yang memiliki rasio keliling terkecil terhadap luasnya
        # menghilangkan kontur dengan bentuk tidak beraturan
        min_ratio = min([cv2.arcLength(cnt, True) / cv2.contourArea(cnt) for cnt in _large_area_contours])
        max_cnt = [cnt for cnt in _large_area_contours
                   if cv2.arcLength(cnt, True) / cv2.contourArea(cnt) == min_ratio][0]

        # gambar semua kontur
        mask_cnt_all = np.zeros((_image.shape[0], _image.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask_cnt_all, contours, -1, 255, thickness=cv2.FILLED)

        # gambar yang diekstrak sebelumnya
        mask_cnt = np.zeros((_image.shape[0], _image.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask_cnt, [max_cnt], -1, 255, thickness=cv2.FILLED)

        # melakukan beberapa operasi morfologi untuk menghilangkan daerah kecil
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask_cnt = cv2.erode(mask_cnt, kernel, mask_cnt, iterations=1)

        # sekali lagi temukan kontur dan ekstrak satu dengan area terluas
        contours, hierarchy = cv2.findContours(mask_cnt, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        max_cnt = max(contours, key=cv2.contourArea)

        # melakukan operasi convextHull yang mengisi lubang di dalam bentuk yang rusak
        mask_covex = np.zeros((_image.shape[0], _image.shape[1]), dtype=np.uint8)
        max_cnt_convex = cv2.convexHull(max_cnt)
        cv2.drawContours(mask_covex, [max_cnt_convex], -1, 255, thickness=cv2.FILLED)
        shape = self._detect_shape_contours_approx(max_cnt_convex)

        # menampilkan hasil
        if self.verbose == 1:
            self.display_contours_restuls(_image, mask_cnt_all, mask_cnt, mask_covex, shape)
        else:
            print("{}: {}".format(self.image_path, shape))
        return None

    def color_based_segmentation(self, image):
        # mengonversi gambar ke format yang berbeda
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # membuat masks untuk warna menggunakan metode inRange
        green_mask = cv2.inRange(hsv_frame, self.colors["low_green"], self.colors["high_green"])
        yellow_mask = cv2.inRange(hsv_frame, self.colors["low_yellow"], self.colors["high_yellow"])

        # melakukan operasi bitwise untuk mengekstrak wilayah dari gambar yang mengandung warna tertentu
        green = cv2.bitwise_and(frame, frame, mask=green_mask)
        yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)
        _width = frame.shape[0]
        _height = frame.shape[1]
        _frame_area = _width * _height

        # method untuk mengembalikan kontur dari wilayah tersegmentasi
        def _return_contours(_segment):
            _gray = cv2.cvtColor(_segment, cv2.COLOR_BGR2GRAY)
            _ret, _threshold = cv2.threshold(_gray, 80, 255, cv2.THRESH_BINARY)
            _contours, _hierarchy = cv2.findContours(_threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            return _contours

        # gabungkan semua kontur yang ditemukan
        cnt_to_consider = []
        for mask in [yellow, green]:
            cnt_to_consider = [*cnt_to_consider,
                               *[cnt for cnt in _return_contours(mask) if cv2.contourArea(cnt) > _frame_area * 0.01]]

        # periksa apakah bentuk gambar memberitahu bentuk bangun
        if _width > _height and (_height / _width) < 0.7 or _width < _height and (_width / _height) < 0.7:
            # jika demikian, maka langsung prediksi kotaknya
            params = self.box_prediction(cnt_to_consider)
            if self.verbose == 1:
                self.display_box_results(image, cnt_to_consider, params)
            else:
                if len(params) > 0:
                    print("{}: {}".format(self.image_path, params[4]))

        # jika tidak, kita perlu memproses kontur lagi untuk prediksi bentuk
        else:
            self._countors_based_segmentation(frame, cnt_to_consider, flag=1)

    def _detect_shape_contours_approx(self, cnt):
        # menemukan simpul dari kontur menggunakan fungsi buildin openCV
        _perimeter = cv2.arcLength(cnt, True)
        _vertices = cv2.approxPolyDP(cnt, 0.012 * _perimeter, True)
        shape = "unknown"

        # mengklasifikasikan bentuk berdasarkan jumlah simpul
        if len(_vertices) == 3:
            shape = "triangle"
        elif len(_vertices) == 4:
            # jika bentuk memiliki empat simpul, maka bisa jadi
            # 1. Persegi
            # 2. Persegi panjang vertikal
            # 3. Persegi panjang horizontal
            # 4. Diamond
            (x, y, w, h) = cv2.boundingRect(_vertices)
            if h != 0:
                _ratio = w / float(h)
                # persegi memiliki rasio aspek ~ 1,0
                if 0.85 <= _ratio <= 1.15:
                    _v_diff_1 = abs(_vertices[0][0][1] - _vertices[1][0][1])
                    _v_diff_2 = abs(_vertices[0][0][1] - _vertices[3][0][1])
                    _thresh = h * 0.1
                    if _v_diff_1 > _thresh and _v_diff_2 > _thresh:
                        shape = "Diamond"
                    else:
                        shape = "Persegi"
                else:
                    shape = "Persegi Panjang Horizontal" if w > h else "Persegi Panjang Vertikal"
        elif len(_vertices) == 5 or len(_vertices) == 6:
            shape = "Segi Lima"
        elif len(_vertices) == 7:
            shape = "Segi Tujuh"
        elif len(_vertices) == 8:
            shape = "Segi Delapan"
        else:
            shape = "Lingkaran"
        return shape

    def box_prediction(self, contours):
        boxes = []
        box_params = []
        # menghitung kotak untuk semua kontur
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])

        # kemudian menggabungkan semua kotak kecil untuk memprediksi satu kotak besar
        if len(boxes) > 0:
            boxes = np.asarray(boxes)
            left, top = np.min(boxes, axis=0)[:2]
            right, bottom = np.max(boxes, axis=0)[2:]
            _box_width = abs(right - left)
            _box_height = abs(bottom - top)

            _ratio = _box_width / _box_height
            if 0.90 <= _ratio <= 1.10:
                shape = 'square'
            else:
                shape = 'Persegi Panjang Vertikal' if _box_width < _box_height else 'Persegi Panjang Horizontal'
            box_params = [left, top, right, bottom, shape]
        return box_params

    def display_contours_restuls(self, image, cnts, extracted_cnts, processed_cnt, shape):
        # tampilkan semua gambar
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))
        fig.suptitle('Bangun Terdeteksi: {}'.format(shape), fontsize=16)
        axes[0].imshow(image)
        axes[0].title.set_text("Gambar Terkuantisasi(KMeans)")
        axes[1].imshow(gray, cmap='gray')
        axes[1].title.set_text("Gambar Skala Abu-abu")
        axes[2].imshow(cnts, cmap='gray')
        axes[2].title.set_text("Kontur Area Besar")
        axes[3].imshow(extracted_cnts, cmap='gray')
        axes[3].title.set_text("Kontur Tunggal Diekstraksi")
        axes[4].imshow(processed_cnt, cmap='gray')
        axes[4].title.set_text("Lubang yang Diisi")
        plt.show()

    def display_box_results(self, image, contours, params):
        # tampilkan semua gambar
        if len(params) > 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            mask_contours = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.drawContours(mask_contours, contours, -1, 255, thickness=1)

            mask_box = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask_box = cv2.rectangle(mask_box, (params[0], params[1]), (params[2], params[3]), 255, thickness=-1)

            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
            fig.suptitle('Bangun Terdeteksi: {}'.format(params[4]), fontsize=16)
            axes[0].imshow(image)
            axes[0].title.set_text("Gambar Terkuantisasi(KMeans)")
            axes[1].imshow(gray, cmap='gray')
            axes[1].title.set_text("Gambar Skala Abu-abu")
            axes[2].imshow(mask_contours, cmap='gray')
            axes[2].title.set_text("Kontur Terdeteksi")
            axes[3].imshow(mask_box, cmap='gray')
            axes[3].title.set_text("Prediksi Bounding Box")
            plt.show()

    def preprcess_image(self, _image, n=8):
        (h, w) = _image.shape[:2]
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2LAB)
        _image = _image.reshape((_image.shape[0] * _image.shape[1], 3))
        clt = MiniBatchKMeans(n_clusters=n)
        labels = clt.fit_predict(_image)
        quant = clt.cluster_centers_.astype("uint8")[labels]
        quant = quant.reshape((h, w, 3))
        _image = _image.reshape((h, w, 3))
        # mengkonversi dari L*a*b* ke RGB
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        return quant

    def predict_shape(self, image_path):
        self.image_path = image_path
        # baca gambarnya
        _image = cv2.imread(image_path)
        # praproses gambar
        _image_quantized = self.preprcess_image(_image)

        # mengubah gambar menjadi skala abu-abu dan memberi efek blur menggunakan Gaussian blur
        _image_grey = cv2.cvtColor(_image_quantized, cv2.COLOR_BGR2GRAY)
        _image_grey = cv2.GaussianBlur(_image_grey, (3, 3), 0)
        # melakukan deteksi canny edge
        _edges = cv2.Canny(_image_grey, 50, 200, None, 3)

        # mendeteksi kontur menggunakan fungsi bawaan
        contours, hierarchy = cv2.findContours(_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnt_closed = [cnt for i, cnt in enumerate(contours) if hierarchy[0][i][2] != -1]
        # lewati kontur untuk diproses lebih lanjut
        self._countors_based_segmentation(_image, cnt_closed, flag=0)
