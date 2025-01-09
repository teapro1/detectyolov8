from PyQt5 import QtGui
import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from ultralytics import YOLO
from gui import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.Button_start.clicked.connect(self.start_capture_video)
        self.uic.Button_stop.clicked.connect(self.stop_capture_video)
        self.uic.Button_pause.clicked.connect(self.pause_resume_capture)
        self.thread = None
        self.paused = False

    def stop_capture_video(self):
        if self.thread:
            self.thread.stop_app()
            self.thread.wait()
        QApplication.quit()

    def start_capture_video(self):
        source_type = self.uic.comboBox.currentText()

        if source_type == "Webcam":
            self.thread = WebcamThread()
        elif source_type == "Image":
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png)")
            if file_path:
                self.thread = ImageThread(file_path)
        elif source_type == "Video":
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
            if file_path:
                self.thread = VideoThread(file_path)

        if self.thread:
            self.thread.signal_total.connect(self.update_total_label)
            self.thread.signal_empty.connect(self.update_empty_label)
            self.thread.signal_occupied.connect(self.update_occupied_label)
            self.thread.signal.connect(self.show_webcam)
            self.thread.signal_full.connect(self.show_full_parking_message)
            self.thread.start()
        else:
            QMessageBox.warning(self, "Error", "No valid source selected.")

    def pause_resume_capture(self):
        if self.thread:
            if self.paused:
                self.thread.resume_capture()
                self.uic.Button_pause.setText("Pause")
            else:
                self.thread.pause_capture()
                self.uic.Button_pause.setText("Resume")
            self.paused = not self.paused

    def update_total_label(self, total):
        self.uic.label_7.setText(str(total))

    def update_empty_label(self, empty_count):
        self.uic.label_8.setText(str(empty_count))

    def update_occupied_label(self, occupied_count):
        self.uic.label_9.setText(str(occupied_count))

    def show_webcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        frame_height, frame_width, _ = cv_img.shape

        # Tính toán kích thước vùng cắt
        x1 = 0
        x2 = frame_width
        y1 = 0
        y2 = frame_height

        # Cắt khung hình theo vùng
        frame_cut = cv_img[y1:y2, x1:x2]
        frame_resized = cv2.resize(frame_cut, (self.uic.label.width(), self.uic.label.height()))
        qt_img = convert_cv_qt(frame_resized)
        self.uic.label.setScaledContents(True)
        self.uic.label.setPixmap(qt_img)

    def show_full_parking_message(self):
        if self.thread:
            self.thread.pause_capture()  # Tạm dừng video hoặc webcam
        QMessageBox.warning(self, "Parking Full", "Bãi đỗ đã đầy. Không thể thêm xe vào nữa.")

def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = QPixmap.fromImage(convert_to_Qt_format)
    return p

class CaptureThread(QThread):
    signal = pyqtSignal(np.ndarray)
    signal_total = pyqtSignal(int)
    signal_empty = pyqtSignal(int)
    signal_occupied = pyqtSignal(int)
    signal_full = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.stop_ = False
        self.pause_ = False
        self.lock = QMutex()

    def pause_capture(self):
        self.lock.lock()
        self.pause_ = True
        self.lock.unlock()

    def resume_capture(self):
        self.lock.lock()
        self.pause_ = False
        self.lock.unlock()

    def run(self):
        pass

    def stop_app(self):
        self.stop_ = True

class WebcamThread(CaptureThread):
    def run(self):
        cap = cv2.VideoCapture(1)  # dùng webcam thì 0, cam ngoài thì 1
        if not cap.isOpened():
            print("Lỗi khi mở camera")
            return

        model = YOLO('teabest.pt')
        total_spots = 50

        while not self.stop_:
            self.lock.lock()
            if not self.pause_:
                self.lock.unlock()
                success, frame = cap.read()
                if not success:
                    print("Lỗi khi kết xuất video!")
                    break
                frame_height, frame_width, _ = frame.shape
                x1 = 0
                x2 = frame_width
                y1 = 0
                y2 = frame_height
                frame_cut = frame[y1:y2, x1:x2]

                # Điều chỉnh ngưỡng phát hiện bằng tham số `conf`
                results = model.track(frame_cut, persist=True, classes=2)

                empty_count = total_spots
                occupied_count = 0

                for result in results:
                    if result:
                        boxes = result.boxes.cpu().numpy()
                        for box in boxes:
                            occupied_count += 1
                            empty_count -= 1
                            x11 = int(box.xyxy[0][0]) + x1
                            y11 = int(box.xyxy[0][1]) + y1
                            x21 = int(box.xyxy[0][2]) + x1
                            y21 = int(box.xyxy[0][3]) + y1
                            frame = cv2.rectangle(frame, (x11, y11), (x21, y21), (255, 0, 0), 2)
                            location = (x11, y11 - 5)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 0.5
                            if box.id:
                                text = "ID: " + str(int(box.id[0]))
                                frame = cv2.putText(frame, text, location, font, fontScale, (0, 0, 255), 2, cv2.LINE_AA)

                if occupied_count > total_spots:
                    self.signal_full.emit()  # Thông báo khi bãi đầy
                    occupied_count = total_spots
                    empty_count = 0

                self.signal_total.emit(total_spots)
                self.signal_empty.emit(empty_count)
                self.signal_occupied.emit(occupied_count)
                self.signal.emit(frame)
            else:
                self.lock.unlock()

        cap.release()



class ImageThread(CaptureThread):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        model = YOLO('teabest.pt')
        frame = cv2.imread(self.file_path)
        if frame is not None:
            frame_height, frame_width, _ = frame.shape
            x1 = 0
            x2 = frame_width
            y1 = 0
            y2 = frame_height
            frame_cut = frame[y1:y2, x1:x2]
            results = model.track(frame_cut, persist=True, classes=2)
            total_spots = 50  # tong so vi tri do
            empty_count = total_spots
            occupied_count = 0

            for result in results:
                if result:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        occupied_count += 1
                        empty_count -= 1
                        x11 = int(box.xyxy[0][0]) + x1
                        y11 = int(box.xyxy[0][1]) + y1
                        x21 = int(box.xyxy[0][2]) + x1
                        y21 = int(box.xyxy[0][3]) + y1
                        # ve khung
                        frame = cv2.rectangle(frame, (x11, y11), (x21, y21), (255, 0, 0), 2)
                        # ve id
                        location = (x11, y11 - 5)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.5
                        if box.id:
                            text = "ID: " + str(int(box.id[0]))
                            frame = cv2.putText(frame, text, location, font, fontScale, (0, 0, 255), 2, cv2.LINE_AA)

            if occupied_count > total_spots:
                self.signal_full.emit()  # Thông báo khi bãi đầy
                occupied_count = total_spots
                empty_count = 0

            self.signal_total.emit(total_spots)
            self.signal_empty.emit(empty_count)
            self.signal_occupied.emit(occupied_count)
            self.signal.emit(frame)


class VideoThread(CaptureThread):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            print("Error: Cannot open video")
            return

        model = YOLO('teabest.pt')
        total_spots = 50  # tong so vi tri do

        while not self.stop_:
            self.lock.lock()
            if not self.pause_:
                self.lock.unlock()
                success, frame = cap.read()
                if not success:
                    print("Lỗi khi kết xuất video")
                    break
                frame_height, frame_width, _ = frame.shape
                x1 = 0
                x2 = frame_width
                y1 = 0
                y2 = frame_height
                frame_cut = frame[y1:y2, x1:x2]
                results = model.track(frame_cut, persist=True,classes=2)
                empty_count = total_spots
                occupied_count = 0

                for result in results:
                    if result:
                        boxes = result.boxes.cpu().numpy()
                        for box in boxes:
                            occupied_count += 1
                            empty_count -= 1
                            x11 = int(box.xyxy[0][0]) + x1
                            y11 = int(box.xyxy[0][1]) + y1
                            x21 = int(box.xyxy[0][2]) + x1
                            y21 = int(box.xyxy[0][3]) + y1
                            # ve khung
                            frame = cv2.rectangle(frame, (x11, y11), (x21, y21), (255, 0, 0), 2)
                            # ve id
                            location = (x11, y11 - 5)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 0.5
                            if box.id:
                                text = "ID: " + str(int(box.id[0]))
                                frame = cv2.putText(frame, text, location, font, fontScale, (0, 0, 255), 2, cv2.LINE_AA)

                if occupied_count > total_spots:
                    self.signal_full.emit()  # Thông báo khi bãi đầy
                    occupied_count = total_spots
                    empty_count = 0

                self.signal_total.emit(total_spots)
                self.signal_empty.emit(empty_count)
                self.signal_occupied.emit(occupied_count)
                self.signal.emit(frame)
            else:
                self.lock.unlock()

        cap.release()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
