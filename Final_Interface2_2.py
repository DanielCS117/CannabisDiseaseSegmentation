import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageQt
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QSlider, QMessageBox, QHBoxLayout, QVBoxLayout, QFrame
from PyQt6.QtGui import QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

class GuideUserInterface(QWidget):
    def __init__(self, model_path):
        super().__init__()
        self.title = 'Inferencia con U-Net'
        self.left = 0
        self.top = 0
        self.width = 1280
        self.height = 720
        self.model_path = model_path
        self.initUI()
        self.loadModel()
        self.mask_alpha = 0.3
        self.mask_visible = True
        self.predicted_mask = None

    def calculate_class_pixels(self, predicted_masks_np):
        class_pixels = {i: 0 for i in range(6)}
        for pixel_value in np.unique(predicted_masks_np):
            if pixel_value in class_pixels:
                class_pixels[pixel_value] = np.sum(predicted_masks_np == pixel_value)
        total_pixels = np.sum(list(class_pixels.values()))
        return class_pixels, total_pixels

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        main_layout = QVBoxLayout()

        legend_layout = QHBoxLayout()
        self.legends = {
            'Fondo': (0, 0, 0),
            'Plantas Sanas': (0, 255, 0),
            'Botritis Etapa 1': (165, 42, 42),
            'Botritis Etapa 2': (128, 0, 128),
            'Botritis Etapa 3': (255, 165, 0),
            'Deficiencias Nutricionales': (255, 255, 0)
        }

        for name, color in self.legends.items():
            legend_item = QFrame()
            legend_item.setFixedSize(20, 20)
            legend_item.setStyleSheet(f'background-color: rgb({color[0]}, {color[1]}, {color[2]}); border-radius: 10px;')

            legend_label = QLabel(name)
            legend_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

            legend_container = QVBoxLayout()
            legend_container.addWidget(legend_item)
            legend_container.addWidget(legend_label)
            legend_layout.addLayout(legend_container)

        main_layout.addLayout(legend_layout)

        self.image_layout = QHBoxLayout()
        self.image_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.label = QLabel(self)
        self.image_layout.addWidget(self.label)
        main_layout.addLayout(self.image_layout)

        self.label_info = QLabel(self)
        main_layout.addWidget(self.label_info)

        button_layout = QHBoxLayout()
        self.btn = QPushButton('Cargar imagen', self)
        self.btn.clicked.connect(self.loadImage)
        button_layout.addWidget(self.btn)

        self.toggle_button = QPushButton(r'Mostrar/Ocultar', self)
        self.toggle_button.clicked.connect(self.toggle_mask)
        button_layout.addWidget(self.toggle_button)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(30)
        self.slider.valueChanged.connect(self.update_transparency)
        button_layout.addWidget(self.slider)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        self.show()

    def loadModel(self):
        num_classes = 6
        self.model_inference = smp.Unet(
            encoder_name='resnet101',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
        try:
            print("Cargando Modelo...")
            state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
            self.model_inference.load_state_dict(state_dict)
            self.model_inference.eval()
            print("Modelo Cargado")
        except Exception as e:
            error_m = f"Error cargando el modelo: {e}"
            print(error_m)
            QMessageBox.critical(self, 'Error', error_m)

    def updateInfoLabel(self, class_pixels, class_percentages):
        info_text = "Píxeles de Clase:\n"
        for class_name, pixels in class_pixels.items():
            info_text += f"{class_name}: {pixels} pixels\n"
        info_text += "\nPorcentajes de clase:\n"
        for class_name, percentage in class_percentages.items():
            info_text += f"{class_name}: {percentage:.2f}%\n"
        self.label_info.setText(info_text)

    def loadImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Abrir Imagen', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if fileName:
            try:
                print(f'Cargando Imagen: {fileName}')
                image = Image.open(fileName).convert('RGB')
                self.image = image

                preprocess = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                input_image = preprocess(image).unsqueeze(0)
                print('Realizando Predicción')
                with torch.no_grad():
                    output = self.model_inference(input_image)
                predicted_masks = torch.argmax(output, dim=1)
                predicted_masks_np = predicted_masks.squeeze(0).numpy().astype(np.uint8)
                self.predicted_mask = predicted_masks_np

                class_pixels, total_pixels = self.calculate_class_pixels(predicted_masks_np)
                class_names = ['Fondo', 'Plantas Sanas', 'Botritis Etapa 1', 'Botritis Etapa 2', 'Botritis Etapa 3', 'Deficiencias Nutricionales']
                class_percentages = {class_names[key]: value / total_pixels * 100 for key, value in class_pixels.items()}
                self.updateInfoLabel(class_pixels, class_percentages)
                self.overlay_img = self.create_overlay_image(self.image, self.predicted_mask)
                self.display_image_with_mask()
                print('Imagen cargada y procesada exitosamente.')
            except Exception as e:
                print(f'Error cargando imagen: {e}')

    def display_image_with_mask(self):
        try:
            print('Mostrando imagen con máscara...')
            if self.overlay_img is None:
                return
            overlay_img = self.overlay_img.copy()
            overlay_img = overlay_img.convert("RGB")
            overlay_img = overlay_img.resize((640, 480), Image.Resampling.LANCZOS)
            overlay_img_qt = ImageQt.ImageQt(overlay_img)
            pixmap = QPixmap.fromImage(overlay_img_qt)
            self.label.setPixmap(pixmap)
        except Exception as e:
            error_m = f'Error mostrando imagen con Máscara: {e}'
            print(error_m)
            QMessageBox.critical(self, 'Error', error_m)

    def create_overlay_image(self, image, mask):
        try:
            print('Superponiendo máscara...')
            image_resized = image.resize((1024, 1024))
            class_colors = {
                0: (0, 0, 0, 0),  # Fondo
                1: (0, 255, 0, int(self.mask_alpha * 255)),  # Plantas Sanas
                2: (165, 42, 42, int(self.mask_alpha * 255)),  # Botritis Etapa 1
                3: (128, 0, 128, int(self.mask_alpha * 255)),  # Botritis Etapa 2
                4: (255, 165, 0, int(self.mask_alpha * 255)),  # Botritis Etapa 3
                5: (255, 255, 0, int(self.mask_alpha * 255)),  # Deficiencias Nutricionales
            }
            mask_img = Image.new('RGBA', (1024, 1024))
            mask_pixels = mask_img.load()
            for y in range(1024):
                for x in range(1024):
                    mask_pixels[x, y] = class_colors[mask[y, x]]
            overlay_img = Image.alpha_composite(image_resized.convert("RGBA"), mask_img)
            return overlay_img
        except Exception as e:
            error_m = f'Error superponiendo máscara: {e}'
            print(error_m)
            QMessageBox.critical(self, 'Error', error_m)
            return image

    def toggle_mask(self):
        try:
            print('Cambiando transparencia de la máscara...')
            self.mask_visible = not self.mask_visible
            if self.mask_visible:
                self.display_image_with_mask()
            else:
                self.display_image(self.image)
        except Exception as e:
            error_m = f'Error cambiando transparencia de la máscara: {e}'
            print(error_m)
            QMessageBox.critical(self, 'Error', error_m)

    def update_transparency(self, value):
        try:
            self.mask_alpha = value / 100
            self.overlay_img = self.create_overlay_image(self.image, self.predicted_mask)
            self.display_image_with_mask()
        except Exception as e:
            error_m = f'Error actualizando transparencia: {e}'
            print(error_m)
            QMessageBox.critical(self, 'Error', error_m)

    def display_image(self, image):
        try:
            print('Mostrando imagen...')
            image_resized = image.resize((640, 480), Image.Resampling.LANCZOS)
            image_qt = ImageQt.ImageQt(image_resized)
            pixmap = QPixmap.fromImage(image_qt)
            self.label.setPixmap(pixmap)
        except Exception as e:
            error_m = f'Error mostrando imagen: {e}'
            print(error_m)
            QMessageBox.critical(self, 'Error', error_m)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GuideUserInterface('model_Unet__41_checkpoint_epoch_40.pt')
    sys.exit(app.exec())
