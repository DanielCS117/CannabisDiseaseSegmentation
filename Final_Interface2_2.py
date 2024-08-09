import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageQt
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QSlider, QMessageBox, QHBoxLayout, QVBoxLayout, QFrame, QScrollArea
from PyQt6.QtGui import QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import psutil

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
        self.zoom_level = 1.0  # Default zoom level

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

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedSize(600, 600) 
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        center_layout = QHBoxLayout()
        center_layout.addStretch(1)
        center_layout.addWidget(self.scroll_area)
        center_layout.addStretch(1)
        main_layout.addLayout(center_layout)

        self.label_info = QLabel(self)
        main_layout.addWidget(self.label_info)

        button_layout = QHBoxLayout()
        self.btn = QPushButton('Cargar imagen', self)
        self.btn.clicked.connect(self.loadImage)
        button_layout.addWidget(self.btn)

        self.toggle_button = QPushButton(r'Mostrar/Ocultar', self)
        self.toggle_button.clicked.connect(self.toggle_mask)
        button_layout.addWidget(self.toggle_button)

        sliders_layout = QVBoxLayout()

        self.mask_transparency_label = QLabel('Mask Transparency', self)
        sliders_layout.addWidget(self.mask_transparency_label)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(30)
        self.slider.valueChanged.connect(self.update_transparency)
        sliders_layout.addWidget(self.slider)

        self.transparency_value_label = QLabel('30%', self) 
        sliders_layout.addWidget(self.transparency_value_label)

        self.zoom_label = QLabel('Zoom', self)
        sliders_layout.addWidget(self.zoom_label)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        sliders_layout.addWidget(self.zoom_slider)

        button_layout.addLayout(sliders_layout)
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
        
        
        info_text += f"\nTransparencia de la máscara: {self.mask_alpha * 100:.0f}%\n"
        info_text += f"Nivel de Zoom: {self.zoom_level * 100:.0f}%\n"

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
                self.class_pixels = class_pixels
                self.class_percentages = class_percentages
                self.updateInfoLabel(self.class_pixels, self.class_percentages)
                print('Imagen cargada y procesada exitosamente.')
            except Exception as e:
                print(f'Error cargando imagen: {e}')

    def display_image_with_mask(self):
        try:
            print(f'Uso de memoria: {psutil.virtual_memory().percent}%')
            print('Mostrando imagen con máscara...')
            if self.overlay_img is None:
                return
            overlay_img = self.overlay_img.copy()
            overlay_img = overlay_img.convert("RGB")
            new_size = (int(1024 * self.zoom_level), int(1024 * self.zoom_level))
            print(f'Nuevo Tamaño de imagen: {new_size}')
            overlay_img = overlay_img.resize(new_size, Image.Resampling.LANCZOS)
            overlay_img_qt = ImageQt.ImageQt(overlay_img)
            pixmap = QPixmap.fromImage(overlay_img_qt)
            self.image_label.setPixmap(pixmap)
            self.image_label.resize(pixmap.size())
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
                5: (255, 255, 0, int(self.mask_alpha * 255))   # Deficiencias Nutricionales
            }
            overlay = Image.new('RGBA', image_resized.size)
            for class_id, color in class_colors.items():
                mask_class = (mask == class_id)
                draw = ImageDraw.Draw(overlay)
                draw.bitmap((0, 0), Image.fromarray((mask_class * 255).astype(np.uint8), mode='L'), fill=color)
            combined = Image.alpha_composite(image_resized.convert('RGBA'), overlay)
            return combined.convert('RGB')
        except Exception as e:
            error_m = f'Error creando imagen superpuesta: {e}'
            print(error_m)
            return None

    def toggle_mask(self):
        try:
            self.mask_visible = not self.mask_visible
            if self.mask_visible:
                self.display_image_with_mask()
            else:
                self.display_image(self.image)
        except Exception as e:
            error_m = f'Error al alternar la máscara: {e}'
            print(error_m)
            QMessageBox.critical(self, 'Error', error_m)

    def update_transparency(self):
        try:
            self.mask_alpha = self.slider.value() / 100.0

            self.transparency_value_label.setText(f'{int(self.mask_alpha * 100)}%')
            
            print(f'Transparencia actualizada: {self.mask_alpha * 100}%')

            if self.image is not None and self.predicted_mask is not None:
                self.overlay_img = self.create_overlay_image(self.image, self.predicted_mask)
                self.display_image_with_mask()
            self.transparency_value_label.repaint()
            
        except Exception as e:
            error_m = f'Error actualizando transparencia: {e}'
            print(error_m)
            QMessageBox.critical(self, 'Error', error_m)



    def update_zoom(self, value):
        self.zoom_level = value / 100.0
        self.display_image_with_mask()
        
        
        self.updateInfoLabel(self.class_pixels, self.class_percentages)


    def display_image(self, image):
        try:
            print('Mostrando imagen...')
            new_size = (int(1024 * self.zoom_level), int(1024 * self.zoom_level))
            image_resized = image.resize(new_size, Image.Resampling.LANCZOS)
            image_qt = ImageQt.ImageQt(image_resized)
            pixmap = QPixmap.fromImage(image_qt)
            self.image_label.setPixmap(pixmap)
            self.image_label.resize(pixmap.size())
        except Exception as e:
            error_m = f'Error mostrando imagen: {e}'
            print(error_m)
            QMessageBox.critical(self, 'Error', error_m)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GuideUserInterface('model_Unet__41_checkpoint_epoch_40.pt')
    sys.exit(app.exec())