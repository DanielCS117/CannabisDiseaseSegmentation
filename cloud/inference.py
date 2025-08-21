import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

class CannabisSegmentationModel:
    def __init__(self, model_filename='model_Unet__46_checkpoint_epoch_40_v2.pt', device='cpu'):
        self.device = device
        self.num_classes = 6

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(BASE_DIR, model_filename)
        self.load_model()

    def load_model(self):
        self.model = smp.Unet(
            encoder_name='resnet101',
            encoder_weights='imagenet',
            in_channels=3,
            classes=self.num_classes
        )
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def preprocess_image(self, image):
        """Recibe un PIL.Image y retorna tensor listo para la inferencia"""
        preprocess = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return preprocess(image).unsqueeze(0).to(self.device)

    def predict_mask(self, image):
        """Recibe PIL.Image, retorna numpy array con clase predicha por pixel"""
        input_tensor = self.preprocess_image(image)
        with torch.no_grad():
            output = self.model(input_tensor)
        predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return predicted_mask

    def calculate_class_metrics(self, mask_np):
        """Retorna cantidad de pixeles por clase y porcentaje"""
        class_pixels = {i: int(np.sum(mask_np == i)) for i in range(self.num_classes)}
        total_pixels = sum(class_pixels.values())
        class_percentages = {i: (v / total_pixels * 100) for i, v in class_pixels.items()}
        return class_pixels, class_percentages

# Función de conveniencia si quieres procesar desde ruta de imagen
def process_image(model_path, image_path):
    image = Image.open(image_path).convert('RGB')
    model = CannabisSegmentationModel(model_path)
    mask = model.predict_mask(image)
    class_pixels, class_percentages = model.calculate_class_metrics(mask)
    return mask, class_pixels, class_percentages

if __name__ == "__main__":
    # Test rápido
    mask, pixels, percentages = process_image('test_image.jpg')
    print("Class pixels:", pixels)
    print("Class percentages:", percentages)
