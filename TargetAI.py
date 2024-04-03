import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QShortcut, QLabel, QLineEdit, QPushButton, QFileDialog, QListWidget,QScrollBar,QCheckBox,QComboBox,QRadioButton
from PyQt5.QtGui import QPixmap, QImage, QFont,QPainter,QPen,QKeySequence
from PyQt5.QtCore import QTimer, Qt, QPoint, QRect
import os
import glob
import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import pathlib
import xml.etree.ElementTree as ET

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

class XMLCreator:
    def __init__(self, save_folder):
        self.save_folder = save_folder

    def create_xml(self, file_name, object_list,height_n,width_n):
        annotation = ET.Element("annotation")

        folder = ET.SubElement(annotation, "folder")
        folder.text = self.save_folder

        filename = ET.SubElement(annotation, "filename")
        filename.text = file_name

        path = ET.SubElement(annotation, "path")
        path.text = os.path.join(self.save_folder, file_name)

        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"

        size = ET.SubElement(annotation, "size")

        height = ET.SubElement(size, "height")
        height.text = height_n

        width = ET.SubElement(size, "width")
        width.text = width_n

        depth = ET.SubElement(size, "depth")
        depth.text = "3"

        for obj in object_list:
            obj_elem = ET.SubElement(annotation, "object")
            name = ET.SubElement(obj_elem, "name")
            name.text = obj["name"]
            pose = ET.SubElement(obj_elem, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(obj_elem, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(obj_elem, "difficult")
            difficult.text = "0"
            bndbox = ET.SubElement(obj_elem, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(obj["xmin"])
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(obj["ymin"])
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(obj["xmax"])
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(obj["ymax"])

        tree = ET.ElementTree(annotation)
        xml_path = os.path.join(self.save_folder, f"{file_name.split('.')[0]}.xml")
        tree.write(xml_path)

class Form(QWidget):

    def __init__(self):
        super().__init__()
        widget = QWidget()

        self.setWindowTitle("TargetAI V1.0.1")
        #self.setGeometry(100, 100, 700, 650)
        self.setFixedSize(1715, 760)  # Form ekranının boyutunu sabitliyoruz

        ## Form Sol Tarafa Dizayn

        # Butonlar
        self.start_button = QPushButton(self)
        self.start_button.setText("Başla (CTRL+A)")
        self.start_button.setGeometry(5, 5,200, 50)
        self.start_button.setEnabled(False)

        self.pause_button = QPushButton(self)
        self.pause_button.setText("Bekle")
        self.pause_button.setGeometry(5, 65,200, 50)
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pause_video)

        self.delete_button = QPushButton(self)
        self.delete_button.setText("Sil (CTRL+D)")
        self.delete_button.setGeometry(5, 125,200, 50)
        self.delete_button.setEnabled(False)

        self.save_button = QPushButton(self)
        self.save_button.setText("Kaydet (CTRL+S)")
        self.save_button.setGeometry(5, 185,200, 50)
        self.save_button.setEnabled(False)

        self.video_button = QPushButton(self)
        self.video_button.setText("Video Seç")
        self.video_button.setGeometry(5, 245, 200, 50)

        # OpenCV HSV renk filtre

        self.openCV_label = QLabel('Arial font',self)
        self.openCV_label.setText("HSV Renk Fitresi   ")
        self.openCV_label.move(10, 320)
        self.openCV_label.setFont(QFont('Arial', 9))

        self.hsv_checkbox = QCheckBox(self)
        self.hsv_checkbox.setText('OpenCV HSV')
        self.hsv_checkbox.setGeometry(15, 340,200, 30)
        self.hsv_checkbox.setChecked(False)  # Başlangıçta seçili olmasın
        self.hsv_checkbox.stateChanged.connect(self.hsv_checkbox_state_changed)

        self.h_low_textbox = QLineEdit(self)
        self.h_low_textbox.setGeometry(70, 380,100,25)
        self.h_low_textbox.setEnabled(False)

        self.h_low_label = QLabel('Arial font',self)
        self.h_low_label.setText("H Low :")
        self.h_low_label.move(10, 380)
        self.h_low_label.setFont(QFont('Arial', 9))

        self.s_low_textbox = QLineEdit(self)
        self.s_low_textbox.setGeometry(70, 410,100,25)
        self.s_low_textbox.setEnabled(False)

        self.s_low_label = QLabel('Arial font',self)
        self.s_low_label.setText("S Low :")
        self.s_low_label.move(10, 410)
        self.s_low_label.setFont(QFont('Arial', 9))

        self.v_low_textbox = QLineEdit(self)
        self.v_low_textbox.setGeometry(70, 440,100,25)
        self.v_low_textbox.setEnabled(False)

        self.v_low_label = QLabel('Arial font',self)
        self.v_low_label.setText("V Low :")
        self.v_low_label.move(10, 440)
        self.v_low_label.setFont(QFont('Arial', 9))

        self.h_up_textbox = QLineEdit(self)
        self.h_up_textbox.setGeometry(70, 500,100,25)
        self.h_up_textbox.setEnabled(False)

        self.h_up_label = QLabel('Arial font',self)
        self.h_up_label.setText("H Up :")
        self.h_up_label.move(10, 500)
        self.h_up_label.setFont(QFont('Arial', 9))

        self.s_up_textbox = QLineEdit(self)
        self.s_up_textbox.setGeometry(70, 530,100,25)
        self.s_up_textbox.setEnabled(False)

        self.s_up_label = QLabel('Arial font',self)
        self.s_up_label.setText("S Up :")
        self.s_up_label.move(10, 530)
        self.s_up_label.setFont(QFont('Arial', 9))

        self.v_up_textbox = QLineEdit(self)
        self.v_up_textbox.setGeometry(70, 560,100,25)
        self.v_up_textbox.setEnabled(False)

        self.v_up_label = QLabel('Arial font',self)
        self.v_up_label.setText("V Up :")
        self.v_up_label.move(10, 560)
        self.v_up_label.setFont(QFont('Arial', 9))

        # Video oynan label
        self.video_label = QLabel(self)
        self.video_label.setGeometry(215, 5,1280,720)

        self.remaining_time_label = QLabel('Arial font',self)
        self.remaining_time_label.setText("0:0:0                                                           .")
        self.remaining_time_label.move(10, 730)
        self.remaining_time_label.setFont(QFont('Arial', 8))

        self.etiket_label = QLabel('Arial font',self)
        self.etiket_label.setText("Etiket    ")
        self.etiket_label.move(1505, 55)
        self.etiket_label.setFont(QFont('Arial', 10))

        self.textbox = QLineEdit(self)
        self.textbox.setGeometry(1505, 75,200,25)

        self.target_label = QLabel('Arial font',self)
        self.target_label.setText("Etiket Listesi   ")
        self.target_label.move(1505, 105)
        self.target_label.setFont(QFont('Arial', 10))

        self.listbox = QListWidget(self)
        self.listbox.setGeometry(1505, 125, 200, 350)
        self.listbox.itemClicked.connect(self.secili_elemani_goster)

        # Scrool bar yüzde oranı göstermek için
        self.yuzde_label = QLabel('Arial font',self)
        self.yuzde_label.setText("50")
        self.yuzde_label.setGeometry(215, 730,50,20)
        self.yuzde_label.setFont(QFont('Arial', 10))

        # Scrool bar yüzde oranı ayarlamak için
        self.scrollbar = QScrollBar(Qt.Horizontal, self)
        self.scrollbar.setMinimum(1)  # Minimum değeri ayarla
        self.scrollbar.setMaximum(98)  # Maksimum değeri ayarla
        self.scrollbar.valueChanged.connect(self.update_label_scrollbar)
        self.scrollbar.setGeometry(250, 730,1200,20)

        self.scan_button = QPushButton(self)
        self.scan_button.setText("Yeniden Tara")
        self.scan_button.setGeometry(1505, 490, 200, 75)
        self.scan_button.clicked.connect(self.scan_frame)
        self.scan_button.setEnabled(False)

         # Radio buttons oluştur
        self.radio_button1 = QRadioButton(self)
        self.radio_button1.setText('TF Etiketi Kullan')
        self.radio_button1.setGeometry(1510, 575,200, 30)
        self.radio_button1.setChecked(True)

        self.radio_button2 = QRadioButton(self)
        self.radio_button2.setText('TF ve Manuel')
        self.radio_button2.setGeometry(1510, 610,200, 30)

        self.radio_button3 = QRadioButton(self)
        self.radio_button3.setText('Sadece Manuel')
        self.radio_button3.setGeometry(1510, 645,200, 30)
        self.radio_button3.clicked.connect(self.radioClicked)

        # Video butonuna tıklanınca dosya seçme iletişim kutusunu aç
        self.video_button.clicked.connect(self.select_video)

        # Başlat ve Bekle butonlarına tıklanınca ilgili metodları çağır
        self.start_button.clicked.connect(self.start_video)


        self.delete_button.clicked.connect(self.delete_frame)
        self.save_button.clicked.connect(self.save_file)


        # Kısayol tuşları
        delete_k = QShortcut(QKeySequence('Ctrl+D'), self)
        delete_k.activated.connect(self.k_delete_frame)

        pause_k = QShortcut(QKeySequence('Ctrl+W'), self)
        pause_k.activated.connect(self.k_pause_video)

        save_k = QShortcut(QKeySequence('Ctrl+S'), self)
        save_k.activated.connect(self.k_save_file)

        next_k = QShortcut(QKeySequence.SelectAll, self)
        next_k.activated.connect(self.k_startvideo)

        self.combobox = QComboBox(self)
        self.combobox.setGeometry(1505, 5, 200, 30)

        # COCO API'de tanımlı olan 80 sınıf ismini ComboBox'a ekliyoruz
        coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]

        self.combobox.addItems(coco_classes)



        # Video dosyası yolu
        self.video_path = ""
        self.video_capture = None
        self.video_fps = 0

        # TensorFlow modelini yükle
        self.detection_model = self.load_model(model_name)
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        # Zamanlayıcı
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)



        # Frame sayacı
        self.frame_count = 0
        self.file_count = 0

        #model pixsel ölçüsü
        self.model_size=640

        self.target_frame=0
        self.frame=0
        self.mask_frame=0

        self.object_list = []
        self.select_index=-1
        self.rect = QRect()

    def hsv_checkbox_state_changed(self, state):
            if state ==  Qt.Checked:
                self.h_low_textbox.setEnabled(True)
                self.s_low_textbox.setEnabled(True)
                self.v_low_textbox.setEnabled(True)
                self.h_up_textbox.setEnabled(True)
                self.s_up_textbox.setEnabled(True)
                self.v_up_textbox.setEnabled(True)

            else:
                self.h_low_textbox.setEnabled(False)
                self.s_low_textbox.setEnabled(False)
                self.v_low_textbox.setEnabled(False)
                self.h_up_textbox.setEnabled(False)
                self.s_up_textbox.setEnabled(False)
                self.v_up_textbox.setEnabled(False)

    def k_pause_video(self):
        if self.pause_button.isEnabled():
            self.pause_video()

    def k_delete_frame(self):
        if self.delete_button.isEnabled():
            self.delete_frame()

    def k_save_file(self):
        if self.save_button.isEnabled():
            self.save_file()

    def k_startvideo(self):
        if self.start_button.isEnabled():
            self.start_video()

    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.drag_start_pos = event.pos()
            self.drag_end_pos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.drag_end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        try:

            if self.radio_button3.isChecked() or self.radio_button2.isChecked() and self.start_button.isEnabled():
                if event.button() == Qt.LeftButton:
                    if self.video_label.pixmap() is not None and self.video_label.geometry().contains(event.pos()) and self.textbox.text()!="" :
                        self.drag_end_pos = event.pos()

                        self.rect = QRect(self.drag_start_pos, self.drag_end_pos)
                        x_diff = self.drag_end_pos.x() - self.drag_start_pos.x()
                        y_diff = self.drag_end_pos.y() - self.drag_start_pos.y()
                        if x_diff>0:
                            xmin_pixel=self.drag_start_pos.x()
                            xmax_pixel=self.drag_end_pos.x()
                        else:
                            xmin_pixel=self.drag_end_pos.x()
                            xmax_pixel=self.drag_start_pos.x()
                        if y_diff>0:
                            ymin_pixel=self.drag_start_pos.y()
                            ymax_pixel=self.drag_end_pos.y()
                        else:
                            ymin_pixel=self.drag_end_pos.y()
                            ymax_pixel=self.drag_start_pos.y()
                        self.object_list.append({
                                        "name": self.textbox.text(),
                                        "xmin": xmin_pixel-215,
                                        "ymin": ymin_pixel-5,
                                        "xmax": xmax_pixel-215,
                                        "ymax": ymax_pixel-5
                                    })
                        #print(f"Sürükleme Başlangıç: {self.drag_start_pos}, Sürükleme Sonu: {self.drag_end_pos}, X Farkı: {x_diff}, Y Farkı: {y_diff}")
                        self.frame_ticket(np.copy(self.frame))
        except Exception as e:
            print("Bir hata oluştu:", e)


    def update_label_scrollbar(self, value):
        self.yuzde_label.setText(str(value))

    def radioClicked(self):
        self.object_list = []
        self.scan_frame()


    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Video Seç", "", "Video dosyaları (*.mp4 *.avi *.mkv *.mov)", options=options)
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.video_fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
            self.start_button.setEnabled(True)

    def start_video(self):
        # Kaydedilecek klasörü belirtin
        if (self.textbox.text()==""):
            self.remaining_time_label.setText("Lütfen bir etiket girin!            ")
        else:
            self.klasor = self.textbox.text()
            self.save_folder = self.klasor
            jpg_dosyalari = glob.glob(os.path.join(self.klasor, '*.jpg'))
            self.file_count = len(jpg_dosyalari)
            # XML oluşturucuyu başlat
            self.xml_creator = XMLCreator(self.save_folder)

            os.makedirs(self.save_folder, exist_ok=True)

            self.timer.start()  # Her saniye yenile
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.textbox.setEnabled(False)

    def secili_elemani_goster(self, item):

        index = self.listbox.row(item)
        self.select_index=index
        frame=np.copy(self.frame)
        self.frame_ticket(frame)

        cv2.rectangle(self.target_frame, (self.object_list[index]["xmin"], self.object_list[index]["ymin"]), (self.object_list[index]["xmax"], self.object_list[index]["ymax"]), (0, 0, 255), 1)

        height, width, channel = self.target_frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.target_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        qImg = qImg.rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)

        self.video_label.setPixmap(pixmap.scaled(1280, 720, Qt.KeepAspectRatio))

        print(index)


    def pause_video(self):
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.textbox.setEnabled(True)

    def list_update(self):
        self.listbox.clear()
        text=self.textbox.text()
        i=1

        for dizi in self.object_list:
            self.listbox.addItem(f"{i}-"+text)
            i=i+1

    def delete_frame(self):
        if self.select_index != -1:
            del self.object_list[self.select_index]
            print(self.object_list)

            frame = np.copy(self.frame)
            self.frame_ticket(frame)

        self.select_index= -1

    def scan_frame(self):
        try:
            frame = np.copy(self.frame)
            ret=1
            if ret:
                text_deger=int(self.yuzde_label.text())
                output_dict = self.run_inference_for_single_image(self.detection_model, frame)
                if output_dict:

                    height, width, channel = frame.shape
                    bytesPerLine = 3 * width
                    self.object_list = []

                    for score, class_id, box in zip(output_dict['detection_scores'], output_dict['detection_classes'], output_dict['detection_boxes']):

                        if score >= (text_deger/100) and self.category_index[class_id]['name'] == self.combobox.currentText():

                            class_name = self.category_index[class_id]['name']

                            ymin, xmin, ymax, xmax = box
                            ymin_pixel = int(ymin * height)
                            xmin_pixel = int(xmin * width)
                            ymax_pixel = int(ymax * height)
                            xmax_pixel = int(xmax * width)

                            if (xmax_pixel-xmin_pixel)<self.model_size and (ymax_pixel-ymin_pixel)<self.model_size:
                                print("Person detected with confidence:", score)
                                print("Coordinates: ymin={}, xmin={}, ymax={}, xmax={}".format(ymin_pixel, xmin_pixel, ymax_pixel, xmax_pixel),score)
                                self.object_list.append({
                                    "name": self.textbox.text(),
                                    "xmin": xmin_pixel,
                                    "ymin": ymin_pixel,
                                    "xmax": xmax_pixel,
                                    "ymax": ymax_pixel
                                })
                                if self.radio_button3.isChecked() :
                                    self.object_list=[]

                    self.scan_button.setEnabled(True)
            self.frame_ticket(frame)
        except Exception as e:
            print("Beklenmeyen bir hata oluştu:", e)

    def update_video_frame(self):
        ret, frame = self.video_capture.read()
        self.frame=np.copy(frame)
        if ret:
            text_deger=int(self.yuzde_label.text())
            output_dict = self.run_inference_for_single_image(self.detection_model, frame)

            height, width, channel = frame.shape
            bytesPerLine = 3 * width


            self.object_list = []
            target_control =0
            for score, class_id, box in zip(output_dict['detection_scores'], output_dict['detection_classes'], output_dict['detection_boxes']):

                if score >= (text_deger/100) and self.category_index[class_id]['name'] == self.combobox.currentText():

                    class_name = self.category_index[class_id]['name']

                    ymin, xmin, ymax, xmax = box
                    ymin_pixel = int(ymin * height)
                    xmin_pixel = int(xmin * width)
                    ymax_pixel = int(ymax * height)
                    xmax_pixel = int(xmax * width)

                    if (xmax_pixel-xmin_pixel)<self.model_size and (ymax_pixel-ymin_pixel)<self.model_size:
                        print("Person detected with confidence:", score)
                        print("Coordinates: ymin={}, xmin={}, ymax={}, xmax={}".format(ymin_pixel, xmin_pixel, ymax_pixel, xmax_pixel),score)
                        target_control=1
                        self.object_list.append({
                                    "name": self.textbox.text(),
                                    "xmin": xmin_pixel,
                                    "ymin": ymin_pixel,
                                    "xmax": xmax_pixel,
                                    "ymax": ymax_pixel
                        })
                        if self.radio_button1.isChecked() or self.radio_button2.isChecked():
                            cv2.rectangle(frame, (xmin_pixel, ymin_pixel), (xmax_pixel, ymax_pixel), (0, 255, 0), 1)
                        else:
                            self.object_list=[]

                if target_control==1:
                    if self.hsv_checkbox.isChecked():
                        self.frame_ticket(self.frame)
                    else:
                        self.list_update()
                    self.delete_button.setEnabled(True)
                    self.save_button.setEnabled(True)
                    self.scan_button.setEnabled(True)

                    self.pause_video()

            self.frame_count += 1

        else:
            self.timer.stop()
            self.video_capture.release()
            print("Video bitti.")
        if target_control==0:
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            qImg = qImg.rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.video_label.setPixmap(pixmap.scaled(1280, 720, Qt.KeepAspectRatio))

    def frame_ticket(self,frame):
        if self.hsv_checkbox.isChecked():
            try:

                image = frame
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print("Lütfen resim seçip tekrar deneyin.")
                return

            try:
                # HSV değerlerini al
                self.h_low_textbox.setEnabled(True)
                self.s_low_textbox.setEnabled(True)
                self.v_low_textbox.setEnabled(True)
                self.h_up_textbox.setEnabled(True)
                self.s_up_textbox.setEnabled(True)
                self.v_up_textbox.setEnabled(True)
                h_min = int(self.h_low_textbox.text())
                h_max = int(self.h_up_textbox.text())
                s_min = int(self.s_low_textbox.text())
                s_max = int(self.s_up_textbox.text())
                v_min = int(self.v_low_textbox.text())
                v_max = int(self.v_up_textbox.text())

                lower_color = np.array([h_min, s_min, v_min])
                upper_color = np.array([h_max, s_max, v_max])
                # Renk maskeleme işlemi
                hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv_image, lower_color, upper_color)

                # Maskeleme sonucunu siyah-beyaz yap
                result = np.where(mask > 0, 255, 0).astype(np.uint8)
                self.mask_frame=np.copy(result)
                frame=np.copy(result)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            except Exception as e:
                print("Bir hata oluştu:", e)
        if self.object_list:
            i=0
            for dizi in self.object_list:
                cv2.rectangle(frame, (self.object_list[i]["xmin"], self.object_list[i]["ymin"]), (self.object_list[i]["xmax"], self.object_list[i]["ymax"]), (0, 255, 0), 1)
                i=i+1
            self.delete_button.setEnabled(True)
            self.save_button.setEnabled(True)
        else:
            self.delete_button.setEnabled(False)
            self.save_button.setEnabled(False)

        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        qImg = qImg.rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        self.target_frame=frame
        self.video_label.setPixmap(pixmap.scaled(1280, 720, Qt.KeepAspectRatio))
        self.list_update()

    def save_file(self):
        if self.object_list:
            if self.hsv_checkbox.isChecked():
                frame = self.mask_frame
            else:
                frame = self.frame
            self.file_count += 1
            file_name = f"{self.textbox.text()}_res_{self.file_count}.jpg"
            file_path = os.path.join(self.save_folder, file_name)
            cv2.imwrite(file_path, frame)
            print(f"Görüntü kaydedildi: {file_path}")
            self.xml_creator.create_xml(file_name, self.object_list,"720","1280")

        self.start_video()


    def load_model(self, model_name):
        base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_file = model_name + '.tar.gz'
        model_dir = tf.keras.utils.get_file(
            fname=model_name,
            origin=base_url + model_file,
            untar=True)

        model_dir = pathlib.Path(model_dir)/"saved_model"
        model = tf.saved_model.load(str(model_dir))

        return model

    def run_inference_for_single_image(self, model, image):
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis,...]

        model_fn = model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        if 'detection_masks' in output_dict:
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                      output_dict['detection_masks'], output_dict['detection_boxes'],
                       image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return output_dict

    def show_results_on_frame(self, frame, output_dict):
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            max_boxes_to_draw=0.2,
            line_thickness=2,
            min_score_thresh=0.50,
            skip_labels=False)

    def enterEvent(self, event):
        self.setCursor(Qt.CrossCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    app.exec_()
