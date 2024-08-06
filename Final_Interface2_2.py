import sys
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageQt
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QSlider
from PyQt6.QtGui import QPixmap
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

        self.label = QLabel(self)
        self.label.move(20, 20)
        self.label.resize(640, 480)

        self.label_info = QLabel(self)
        self.label_info.move(50, 350)
        self.label_info.resize(640, 480)

        self.btn = QPushButton('Cargar imagen', self)
        self.btn.move(600, 300)
        self.btn.clicked.connect(self.loadImage)

        self.toggle_button = QPushButton('Toggle Mask', self)
        self.toggle_button.move(800, 300)
        self.toggle_button.clicked.connect(self.toggle_mask)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setGeometry(800, 350, 160, 30)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(30)
        self.slider.valueChanged.connect(self.update_transparency)

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
            print("Loading model...")
            state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
            self.model_inference.load_state_dict(state_dict)
            self.model_inference.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def updateInfoLabel(self, class_pixels, class_percentages):
        info_text = "Píxeles de Clase:\n"
        for class_name, pixels in class_pixels.items():
            info_text += f"{class_name}: {pixels} pixels\n"
        info_text += "\nPorcentajes de clase:\n"
        for class_name, percentage in class_percentages.items():
            info_text += f"{class_name}: {percentage:.2f}%\n"
        self.label_info.setText(info_text)

    def loadImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            try:
                print(f"Loading image: {fileName}")
                image = Image.open(fileName).convert('RGB')
                self.image = image

                preprocess = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                input_image = preprocess(image).unsqueeze(0)
                print("Performing inference...")
                with torch.no_grad():
                    output = self.model_inference(input_image)
                predicted_masks = torch.argmax(output, dim=1)
                predicted_masks_np = predicted_masks.squeeze(0).numpy().astype(np.uint8)
                self.predicted_mask = predicted_masks_np

                class_pixels, total_pixels = self.calculate_class_pixels(predicted_masks_np)
                class_names = ['Fondo', 'Plantas Sanas', 'Botritis Etapa 1', 'Botritis Etapa 2', 'Botritis Etapa 3', 'Deficiencias Nutricionales']
                class_percentages = {class_names[key]: value / total_pixels * 100 for key, value in class_pixels.items()}
                self.updateInfoLabel(class_pixels, class_percentages)

                self.display_image_with_mask(self.image, self.predicted_mask)
                print("Image loaded and processed successfully.")
            except Exception as e:
                print(f"Error loading image: {e}")

    def display_image_with_mask(self, image, mask):
        try:
            print("Displaying image with mask...")
            overlay_img = self.overlay_mask_on_image(image, mask)
            overlay_img = overlay_img.convert("RGB")
            overlay_img = overlay_img.resize((640, 480), Image.Resampling.LANCZOS)
            overlay_img_qt = ImageQt.ImageQt(overlay_img)
            pixmap = QPixmap.fromImage(overlay_img_qt)
            self.label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error displaying image with mask: {e}")

    def overlay_mask_on_image(self, image, mask):
        try:
            print("Overlaying mask on image...")
            image_resized = image.resize((1024, 1024))
            class_colors = {
                0: (0, 0, 0, 0),  # Fondo
                1: (0, 255, 0, int(self.mask_alpha * 255)),  # Plantas Sanas
                2: (165, 42, 42, int(self.mask_alpha * 255)),  # Botritis Etapa 1
                3: (128, 0, 128, int(self.mask_alpha * 255)),  # Botritis Etapa 2
                4: (255, 165, 0, int(self.mask_alpha * 255)),  # Botritis Etapa 3
                5: (255, 255, 0, int(self.mask_alpha * 255)),  # Deficiencias Nutricionales
            }
            mask_img = Image.new("RGBA", (1024, 1024))
            mask_pixels = mask_img.load()
            for y in range(1024):
                for x in range(1024):
                    mask_pixels[x, y] = class_colors[mask[y, x]]
            overlay_img = Image.alpha_composite(image_resized.convert("RGBA"), mask_img)
            return overlay_img
        except Exception as e:
            print(f"Error overlaying mask on image: {e}")
            return image

    def toggle_mask(self):
        try:
            print("Toggling mask visibility...")
            self.mask_visible = not self.mask_visible
            if self.mask_visible:
                self.display_image_with_mask(self.image, self.predicted_mask)
            else:
                self.display_image(self.image)
        except Exception as e:
            print(f"Error toggling mask: {e}")

    def update_transparency(self, value):
        try:
            print(f"Updating transparency: {value}")
            self.mask_alpha = value / 100
            if self.mask_visible:
                self.display_image_with_mask(self.image, self.predicted_mask)
        except Exception as e:
            print(f"Error updating transparency: {e}")

    def display_image(self, image):
        try:
            print("Displaying image without mask...")
            image = image.convert("RGB")
            image = image.resize((640, 480), Image.Resampling.LANCZOS)
            image_qt = ImageQt.ImageQt(image)
            pixmap = QPixmap.fromImage(image_qt)
            self.label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error displaying image: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GuideUserInterface("model_Unet__41_checkpoint_epoch_40.pt")  # Update the path to your model
    sys.exit(app.exec())
