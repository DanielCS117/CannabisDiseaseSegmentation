import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_btn_load_image_Interface(object):
    def setupUi(self, btn_load_image):
        btn_load_image.setObjectName("btn_load_image")
        btn_load_image.resize(1077, 836)
        self.InfoLabel = QtWidgets.QLabel(btn_load_image)
        self.InfoLabel.setGeometry(QtCore.QRect(60, 500, 251, 291))
        self.InfoLabel.setText("")
        self.InfoLabel.setObjectName("InfoLabel")
        self.label = QtWidgets.QLabel(btn_load_image)
        self.label.setGeometry(QtCore.QRect(290, 40, 491, 101))
        font = QtGui.QFont()
        font.setFamily("MS Gothic")
        font.setPointSize(48)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(btn_load_image)
        self.label_2.setGeometry(QtCore.QRect(20, 120, 1041, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(btn_load_image)
        self.label_3.setGeometry(QtCore.QRect(400, 180, 271, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(btn_load_image)
        self.label_4.setGeometry(QtCore.QRect(410, 220, 241, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.retranslateUi(btn_load_image)
        QtCore.QMetaObject.connectSlotsByName(btn_load_image)

    def retranslateUi(self, btn_load_image):
        _translate = QtCore.QCoreApplication.translate
        btn_load_image.setWindowTitle(_translate("btn_load_image", "Dialog"))
        self.label.setText(_translate("btn_load_image", "CANNAVANA S.A.S"))
        self.label_2.setText(_translate("btn_load_image", "SISTEMA DE SEGUIMIENTO DE CRECIMIENTO DE CULTIVO DE CANNABIS UTILIZANDO TECNICAS DE VISION ARTIFICIAL"))
        self.label_3.setText(_translate("btn_load_image", "JOSÉ DANIEL CALA SUÁREZ"))
        self.label_4.setText(_translate("btn_load_image", "INGENIERÍA MECATRÓNICA"))


class GuideUserInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Inferencia con U-Net'
        self.left = 0
        self.top = 0
        self.width = 1280
        self.height = 720  
        self.initUI()
        self.loadModel()

    def calculate_class_pixels(self, predicted_masks_np):
        class_pixels = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # 

        for pixel_value in np.unique(predicted_masks_np):
            if pixel_value in class_pixels:
                class_pixels[pixel_value] = np.sum(predicted_masks_np == pixel_value)

        total_pixels = np.sum(list(class_pixels.values()))

        return class_pixels, total_pixels
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label = QLabel(self)
        self.label.move(20, 20)
        self.label.resize(640, 480)

        self.label_info = QLabel(self)  
        self.label_info.move(50, 350)
        self.label_info.resize(640, 480)

        self.btn = QPushButton('Cargar imagen', self)
        self.btn.move(600, 300)

        self.show()

    
    def loadModel(self):
        num_classes = 6
        self.model_inference = smp.Unet(
            encoder_name='resnet101',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
        state_dict = torch.load("model_Unet__41_checkpoint_epoch_40.pt", map_location=torch.device('cpu'))
        self.model_inference.load_state_dict(state_dict)
        self.model_inference.eval()

    def updateInfoLabel(self, class_pixels, class_percentages):
        info_text = "Píxeles de Clase:\n"
        for class_name, pixels in class_pixels.items():
            info_text += f"{class_name}: {pixels} pixels\n"

        info_text += "\nPorcentajes de clase:\n"
        for class_name, percentage in class_percentages.items():
            info_text += f"{class_name}: {percentage}%\n"

        self.label_info.setText(info_text)

    def loadImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            image = cv2.imread(fileName)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_for_inference = transforms.ToPILImage()(image)

            preprocess = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            preprocess2 = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor()
            ])

            input_image = preprocess(image_for_inference).unsqueeze(0)
            imagen_or = preprocess2(image_for_inference).unsqueeze(0)

            with torch.no_grad():
                output = self.model_inference(input_image)

            predicted_masks = torch.argmax(output, dim=1)

            input_image_np = input_image.squeeze(0).permute(1, 2, 0).numpy()
            imagen_or = imagen_or.squeeze(0).permute(1, 2, 0).numpy()
            predicted_masks_np = predicted_masks.squeeze(0).numpy()

            class_pixels, total_pixels = self.calculate_class_pixels(predicted_masks_np)

            class_names = ['Fondo', 'Plantas Sanas', 'Botritis Etapa 1', 'Botritis Etapa 2', 'Botritis Etapa 3', 'Deficiencias Nutricionales']
            class_colors = ['black', 'lime', 'brown', 'purple', 'orange', 'yellow']

            cmap = plt.cm.colors.ListedColormap(class_colors) 

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
            #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(imagen_or)
            ax1.axis('off')
            ax1.set_title('Imagen Original')

            im = ax2.imshow(predicted_masks_np, cmap=cmap, vmin=0, vmax=len(class_names)-1)
            ax2.axis('off')
            ax2.set_title('Máscara Predicha')

            cbar = plt.colorbar(im, ax=[ax2, ax3], ticks=np.arange(len(class_names)), boundaries=np.arange(len(class_names)+1) -0.5)
            cbar.set_ticklabels(class_names)

            #cbar = plt.colorbar(im, ax=ax2, ticks=np.arange(len(class_names)-1)+1, boundaries=np.arange(len(class_names)-1) + 1)
            #cbar.set_ticklabels(class_names[1:])

            plt.tight_layout()

            class_percentages = {class_names[key]: value / total_pixels * 100 for key, value in class_pixels.items()}
            self.updateInfoLabel(class_pixels, class_percentages)

            plt.show()



class Ui_btn_load_image(Ui_btn_load_image_Interface, GuideUserInterface):
    def setupUi(self, btn_load_image):
        super().setupUi(btn_load_image)
        self.loadModel()
        self.btn.clicked.connect(self.loadImage)

        self.InfoLabel = self.label_info
        self.label = self.label_2

        self.retranslateUi(btn_load_image)
        QtCore.QMetaObject.connectSlotsByName(btn_load_image)

    def retranslateUi(self, btn_load_image):
        _translate = QtCore.QCoreApplication.translate
        btn_load_image.setWindowTitle(_translate("btn_load_image", "Dialog"))
        self.label.setText(_translate("btn_load_image", "CANNAVANA S.A.S"))
        self.label_2.setText(_translate("btn_load_image", "SISTEMA DE SEGUIMIENTO DE CRECIMIENTO DE CULTIVO DE CANNABIS UTILIZANDO TECNICAS DE VISION ARTIFICIAL"))
        self.label_3.setText(_translate("btn_load_image", "JOSÉ DANIEL CALA SUÁREZ"))
        self.label_4.setText(_translate("btn_load_image", "INGENIERÍA MECATRÓNICA"))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    btn_load_image = QWidget()
    ui = Ui_btn_load_image()
    ui.setupUi(btn_load_image)
    btn_load_image.show()
    sys.exit(app.exec_())

