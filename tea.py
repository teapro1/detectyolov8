import time

from PyQt5 import QtGui
import sys
import cv2
import numpy as np
from PyQt5.QtCore import QMutex, QWaitCondition
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from ultralytics import YOLO
from gui import Ui_MainWindow
import serial


# Mở cổng Serial
arduino = serial.Serial('COM8', 9600, timeout=1)

def send_data_to_arduino(data):
    if arduino.isOpen():
        arduino.write(f"{data}\n".encode())
        arduino.flush()
        print(f"Sent to Arduino: {data}")
        time.sleep(0.1)  # Thêm thời gian trễ ngắn

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
        self.uic.label_10.setText("")
        self.uic.label_10.setStyleSheet("")  

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
        self.uic.label_10.setText("Bãi đỗ đã đầy. Không thể thêm xe vào nữa.")
        self.uic.label_10.setStyleSheet("color: red; font-weight: bold;")


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
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.is_running = True
        self.is_paused = False
        self.is_full = False
        self.model = YOLO('teabest.pt')

    def run(self):
        while True:
            self.mutex.lock()
            if not self.is_running:
                self.mutex.unlock()
                break
            if self.is_paused:
                self.wait_condition.wait(self.mutex)
            self.mutex.unlock()

            frame = self.get_frame()
            if frame is not None:
                detected_boxes = self.process_frame(frame)
                self.signal.emit(frame)

                occupied_count = min(len(detected_boxes), 4)
                empty_count = max(0, 4 - occupied_count)

                self.signal_total.emit(4)
                self.signal_occupied.emit(occupied_count)
                self.signal_empty.emit(empty_count)

                send_data_to_arduino(str(occupied_count))
                if occupied_count >= 4 and not self.is_full:
                    self.signal_full.emit()
                    self.is_full = True  # Đánh dấu đã gửi tín hiệu đầy
                elif occupied_count < 4:
                    self.is_full = False  # Đặt lại khi bãi không đầy

    def stop_app(self):
        self.mutex.lock()
        self.is_running = False
        self.is_paused = False
        self.wait_condition.wakeAll()
        self.mutex.unlock()
        self.quit()

    def pause_capture(self):
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()

    def resume_capture(self):
        self.mutex.lock()
        self.is_paused = False
        self.wait_condition.wakeAll()
        self.mutex.unlock()

    def process_frame(self, frame):
        results = self.model(frame, classes=2)
        detected_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0),
                              2)

        return detected_boxes

    def get_frame(self):
        # Implement this method to get a frame from your source
        pass


class WebcamThread(CaptureThread):
    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            self.mutex.lock()
            if not self.is_running:
                self.mutex.unlock()
                break
            if self.is_paused:
                self.wait_condition.wait(self.mutex)
            self.mutex.unlock()

            ret, frame = cap.read()
            if not ret:
                break

            detected_boxes = self.process_frame(frame)
            self.signal.emit(frame)

            occupied_count = min(len(detected_boxes), 4)
            empty_count = max(0, 4 - occupied_count)

            self.signal_total.emit(4)
            self.signal_occupied.emit(occupied_count)
            self.signal_empty.emit(empty_count)
            send_data_to_arduino(str(occupied_count))
            if occupied_count >= 4:
                self.signal_full.emit()
                self.stop_app()

        cap.release()


class VideoThread(CaptureThread):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        cap = cv2.VideoCapture(self.file_path)

        while True:
            self.mutex.lock()
            if not self.is_running:
                self.mutex.unlock()
                break
            if self.is_paused:
                self.wait_condition.wait(self.mutex)
            self.mutex.unlock()

            ret, frame = cap.read()
            if not ret:
                break

            detected_boxes = self.process_frame(frame)
            self.signal.emit(frame)

            occupied_count = min(len(detected_boxes), 4)
            empty_count = max(0, 4 - occupied_count)

            self.signal_total.emit(4)
            self.signal_occupied.emit(occupied_count)
            self.signal_empty.emit(empty_count)
            send_data_to_arduino(str(occupied_count))
            if occupied_count >= 4:
                self.signal_full.emit()
                self.stop_app()

        cap.release()


class ImageThread(CaptureThread):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        frame = cv2.imread(self.file_path)

        detected_boxes = self.process_frame(frame)
        self.signal.emit(frame)

        occupied_count = min(len(detected_boxes), 4)
        empty_count = max(0, 4 - occupied_count)

        self.signal_total.emit(4)
        self.signal_occupied.emit(occupied_count)
        self.signal_empty.emit(empty_count)
        send_data_to_arduino(str(occupied_count))
        if occupied_count >= 4:
            self.signal_full.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
