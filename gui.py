import cv2
import sys
import numpy as np
from PyQt5 import QtWidgets, uic, QtGui

from dog_vs_cat import CAT_INDEX, DOG_INDEX, Net, WIDTH, HEIGHT


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(r'dog_vs_cat.ui', self)
        self.net = Net(WIDTH, HEIGHT)
        self.net.load(r"model")
        

        self.pb_load_image:           QtWidgets.QPushButton
        self.pb_predict:              QtWidgets.QPushButton
        self.label_image:             QtWidgets.QLabel
        self.label_dog:               QtWidgets.QLabel
        self.label_cat:               QtWidgets.QLabel

        self.pb_predict.clicked.connect(self._predict)
        self.pb_load_image.clicked.connect(self._load_image)

        self.current_image: np.ndarray = None

        self.show()

    def _update_image(self, path: str):
        if self.current_image is None:
            return

        pixmap = QtGui.QPixmap(path)
        pixmap = pixmap.scaled(self.label_image.width(), self.label_image.height())
        
        self.label_image.setPixmap(pixmap)

    def _predict(self):
        image = cv2.resize(self.current_image, (WIDTH, HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = self.net.forward(image)[0]

        self.label_cat.setText("Cat: predict Value: {:.3f}".format(res[CAT_INDEX] * 100))
        self.label_dog.setText("Dog: predict Value: {:.3f}".format(res[DOG_INDEX] * 100))

    def _load_image(self):
        res = QtWidgets.QFileDialog.getOpenFileName(self)
        if len(res[0]) > 0:
            self.current_image = cv2.imread(res[0])
            self._update_image(res[0])



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    _ = Ui()
    app.exec_()
