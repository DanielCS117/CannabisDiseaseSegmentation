# Cannabis-Disease Segmentation

**Semantic Segmentation of Cannabis Plant Diseases using Deep Learning (COCO Format)**

## PROJECT OVERVIEW
This project implements a semantic segmentation model to detect 6 classes, 3 of them corresponding to diseases in cannabis plants, 1 related to their nutrition, and support classes for the model operation (healthy plants and background) from general images. It uses a dataset in COCO format, and has a graphical interface that facilitates the visualization of the inference results, superimposing the predicted mask on the evaluated image.

<p align="center">
  <img src="assets/scene.png" width="200"/>
</p>

<p align="center"><em>Figure 1. Cannabis Plant scene</em></p>


### Class Definitions

The model segments cannabis plant images into the following semantic classes:

| ID | Class Name        | Description                          |
|----|-------------------|--------------------------------------|
| 0  | Background         | Non-plant area                       |
| 1  | Healthy Plants       | Cannabis leaves without visible symptoms                       |
| 2  | Botrytis 1       | Early-stage fungal infection (brown spots)            |
| 3  | Botrytis 2      | Moderate Botrytis             |
| 4  | Botrytis 3          | Advanced infection (necrosis, dryness)             |
| 5  | Nutritional Deficiencies          | Yellowing, spotting due to lack of N/P/K                 |

According to the classes, all morphologic disease can be identified as: 

<p align="center">
  <img src="assets/healthy.png" width="150"/>
</p>

<p align="center"><em>Figure 2. Healthy Plant</em></p>     

<p align="center">
  <img src="assets/botritis1.png" width="150"/>
</p>

<p align="center"><em>Figure 3. Botrytis 1</em></p>    

<p align="center">
  <img src="assets/botritis2.png" width="150"/>
</p>

<p align="center"><em>Figure 4. Botrytis 2</em></p>    

<p align="center">
  <img src="assets/botritis3.png" width="150"/>
</p>

<p align="center"><em>Figure 5. Botrytis 3</em></p>    

<p align="center">
  <img src="assets/deficiencies.png" width="150"/>
</p>

<p align="center"><em>Figure 6. Nutritional deficiencies</em></p>    

## 🔁 PROJECT PIPELINE

A --> [Manual Annotation (Supervisely)] --> B[COCO JSON Parsing]  
B --> C[Dataset & Dataloader Definition]  
C --> D[Model Training (UNet + ResNet101)]  
D --> E[Model Saving (.pt file)]  
E --> F[GUI Inference with Mask Overlay] 

🔹 A: Manual Annotation (Supervisely)"

The annotation process was carried out using Supervisely, a tool that allows precise labeling of semantic classes.
Each region of interest (disease symptoms, nutrient issues, healthy areas) was labeled using polygon tools.
Annotations were exported in COCO format for later use in PyTorch pipelines.

<p align="center">
  <img src="assets/partial-label.png" width="350"/>
</p>
<p align="center"><em>Figure 5. Botrytis 3</em></p>    

🔹 B: COCO JSON Parsing
A custom Python class was implemented to load and convert COCO-format annotations into image-mask pairs.
Each pixel in the mask corresponds to a class ID, enabling semantic segmentation training.
The class:

* Parses annotation masks per image into multi-channel tensors (one channel per class):

    ```def _generate_masks(self, annotations, image_size):
        """
        Generates binary masks for all defined classes based on annotations.

        Args:
            annotations (list): List of annotations for a single image.
            image_size (tuple): (width, height) of the image.

        Returns:
            torch.Tensor: Multi-channel mask tensor with one channel per class.
        """
        num_classes = self.num_classes  
        masks = torch.zeros(num_classes, *image_size)  
        category_counts = {category_id: 0 for category_id in range(num_classes)}  
        total_objects = 0  

        for ann in annotations:
            category_id = ann['category_id']
            if category_id in range(0,6):
                mask = self.coco.annToMask(ann)

            # Get binary mask per channel
                mask_tensor = torch.from_numpy(mask).float()
                binary_mask = (mask_tensor > 0).float()
                masks[category_id] += binary_mask
                category_counts[category_id] += 1
                total_objects += 1

        print("Count by category:", category_counts)
        print("Total count:", total_objects)

        return masks

* Computes class frequencies and derives normalized class weights to address imbalance.

    ```def _calculate_class_frequencies(self):
        """
        Calculates the pixel area covered by each class across all images.

        Returns:
            np.ndarray: Array of class frequencies (total area per class).
        """
        class_frequencies = np.zeros(self.num_classes)  

        for image_id in self.image_ids:
            annotation_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(annotation_ids)

            for ann in annotations:
                category_id = ann['category_id']
                if category_id in range(0,6):
                    class_frequencies[category_id] += ann['area']
        
        for category_id, frequency in enumerate(class_frequencies):
          print(f"Category ID: {category_id}, Frequency: {frequency}")

        return class_frequencies

    def _calculate_class_weights(self):
        """
        Computes class weights inversely proportional to class area frequency.

        Returns:
            torch.Tensor: Normalized class weights as a tensor.
        """
        total_samples = np.sum(self.class_frequencies)
        class_areas = np.zeros_like(self.class_frequencies)

        for image_id in self.image_ids:
            annotation_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(annotation_ids)

            for ann in annotations:
                category_id = ann['category_id']
                if category_id in range(0,6):
                    area = ann['area']
                    class_areas[category_id] += area

        class_weights = 1.0 / (6.0*class_areas / total_samples) 
        class_weights /= np.max(class_weights)  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

        return class_weights

* Supports __getitem__, __len__, and a custom collate_fn for batching.

    ```def custom_collate_fn(batch):
    """
    Custom collate function to batch images and masks for DataLoader.

    Args:
        batch (list): List of (image, mask) tuples.

    Returns:
        tuple: Batched images and masks.
    """
    images, masks = zip(*batch)
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)
    else:
        image_transform = transforms.ToTensor()
        images = [image_transform(image) for image in images]
        images = torch.stack(images, dim=0)

    masks = torch.stack(masks, dim=0)
    return images, masks

* The dataset is integrated with PyTorch’s DataLoader for both training and validation, applying:

    * Resizing to 1024×1024,

    * Normalization (ImageNet stats),

    * Conversion to tensor format.

    ```split1 = "train"
    coco_json_file1 = r'Base de Datos\train\annotations\train_data1024.json'
    split2 = "test"
    coco_json_file2 = r'Base de Datos\test\annotations\test_data1024.json'

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
🔹 C: Dataset & Dataloader Definition
The dataset class is compatible with PyTorch and uses Dataset and DataLoader to load mini-batches. This ensures compatibility with the model input.

        # Dataset and DataLoader creation
        dataset_train = CocoSemSegUNET(coco_json_file1, split1, transform=transform)
        dataloader_train = data.DataLoader(dataset_train, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
        dataset_test = CocoSemSegUNET(coco_json_file2, split2, transform=transform)
        dataloader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
        dataset_used = "RAW 6 Classes"

🔹 D: Model Training (UNet + ResNet101)
The training pipeline uses Segmentation Models PyTorch (SMP) with a UNet architecture and ResNet101 backbone.
It includes training loops, loss functions (CrossEntropy and Weighted Crossentropy), metric tracking, and validation.
Weights are saved periodically for best validation performance.

        num_classes = 6
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class_weights_train = torch.tensor(class_weights_train).to(device)
        class_weights_test = torch.tensor(class_weights_test).to(device)

        encoder = 'resnet101'   # Encoder backbone for U-Net
        e_weights = 'imagenet'  # Use pretrained ImageNet weights
        train_mode = 'RAW'      # Custom training label/tag

        # Pre-Trained Model from SEGMENTATION MODELS PYTORCH Library
        model_Unet = smp.Unet(
            encoder_name=f'{encoder}',
            encoder_weights=f'{e_weights}',
            in_channels=3,
            classes=num_classes
        )

        criterion_ce = nn.CrossEntropyLoss()                                # Standard cross-entropy
        criterion_w = nn.CrossEntropyLoss(weight = class_weights_train)     # Weighted cross-entropy

        # Main hyperparameters
        criterion_train = criterion_ce
        criterion = criterion_ce
        learning_rate = 0.0001
        weight_decay = 0.0005
        optimizer = torch.optim.Adam(model_Unet.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_Unet.to(device)
        model_Unet.train()
        print(device)

🔹 E: Model Saving (.pt file)
Trained weights are saved to a .pt file after evaluation.
This file can later be loaded for inference or deployment in other environments (e.g., cloud). To perform inference by loading the weights from the PT file, recreate exactly the same model that was used for training, which in this case is a UNET model downloaded from the SMP library with ResNet101 backbone and initial ImageNet weights.

🔹 F: GUI Inference with Mask Overlay
A graphical user interface built with PyQt6 allows users to load images and perform inference locally.
Predicted masks are overlayed on the original image with adjustable transparency, aiding visual interpretation.

The main functions include methods to calculate the pixels per class (to determine the percentages of regions of each class), a method to initialize the model and load the weights trained for the task, a method to update the prediction labels, another to define correction recommendations, another to load an external image, and others to generate the prediction mask overlay. Of course, there are also the standard methods for manipulating interface parameters (zoom, transparency and buttons) and the general methods for running the application. 

📊 Results: The results shown below comprise the result of visual prediction by parallel comparison and mask overlay through the GUI application. 

✅ Advantages of the chosen approach

Semantic segmentation proved to be the most viable approach for this project compared to image classification and object detection. This is because it allows multiple defects within a single image to be identified with greater spatial accuracy, even in complex scenes where objects exhibit different shapes, sizes, directions and levels of grouping.

The U-Net based model with ResNet-101 encoder showed acceptable visual performance in the classes with the highest representation in the database, particularly in healthy leaves, leaves with stage 1 Botrytis and leaves with nutritional deficiencies. This is evidence of the encoder's ability to extract relevant patterns in these types of problems.

⚠️ Difficulties encountered

The manual labeling process was especially challenging due to the irregular distribution of leaves and diseases in the images. This complexity prevented the effective use of AI-assisted labeling tools such as Supervisely, forcing completely manual and detailed work.

One of the biggest challenges was the precise delineation of contours between classes. The morphological similarity between certain classes under variations in illumination or position generated confusion even at the human level, which likely introduced noise into the training data. For example:

* A nutritionally deficient leaf under shade may resemble one with stage 3 Botrytis.

* A healthy leaf with poor lighting may be mistaken for stage 1 Botrytis because of its darkened coloration.

These ambiguities affected the model's ability to correctly differentiate between similar classes, and contributed to a decrease in metric performance.

📊 Metrics and analysis

* The main metric was IoU (method 1), with results of 0.38 in training and 0.40 in validation.

* Although these values may seem low, they reflect the overall average including poorly represented and difficult to differentiate classes. Despite this, the model achieved a mask similarity of 40% in testing, which is functional for practical cases.

* F1-score metrics (0.84 in training and 0.75 in validation), as well as precision and recall, showed consistent values suggesting acceptable model performance.

* A bias toward the most represented classes was observed, indicating that the model tends to assign most pixels to them in ambiguous situations.

* The use of techniques such as weight decay regularization and inverse class frequency balancing partially improved model performance, but further fine tuning of hyperparameters is required to maximize their impact.



<p align="center">
  <img src="assets/paralelpred46.png" width="350"/>
</p>
<p align="center"><em>Figure. Parallel prediction</em></p>   

<p align="center">
  <img src="assets/app46.png" width="350"/>
</p>
<p align="center"><em>Figure 5. GUI prediction</em></p>    

<p align="center">
  <img src="assets/plots46.png" width="350"/>
</p>
<p align="center"><em>Figure 5. Metrics results</em></p>    

    Puedes incluir algunas capturas de pantalla o imágenes con ejemplos de predicciones. 📸


## REPOSITORY SCTRUCTRE
Root    
├── Final_Interface2_2.py               # Graphic interface   
├── Pytorch_COCO_SEGMENTATION.ipynb     # Training and Inference Notebook  
├── README.md                           # Proyect Documentation    
└── LICENSE                             # Proyect Licence     
 
## 🚀 Features
✅ Custom dataset class converting COCO JSON into (image, mask) pairs labeled from SuperViseLY. 

✅ Batch generation with PyTorch Dataset and DataLoader

✅ UNet model with ResNet101 backbone (SMP)

✅ Interactive GUI to visualize predictions

✅ Model checkpointing and test-time inference

✅ COCO-compliant annotation processing

## 🛠️ Technologies
Python 3.x

PyTorch & torchvision

Segmentation Models PyTorch (SMP)

PyQt6 (for GUI)

Supervisely (for annotation)

COCO JSON format

📌 How to Run
Clone the repo:

git clone https://github.com/DanielCS117/Cannabis-Disease

Install requirements:

pip install -r requirements.txt
Train model (see the notebook).

Run GUI:

python Final_Interface2_2.py

```  import os
import numpy as np
import random
import torch
import torchvision
import torchvision.utils as utils
from torchvision import io, transforms
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import torch.optim as optim
from PIL import Image, ImageOps, ImageFilter
from pycocotools.coco import COCO
from tqdm.auto import tqdm
from pycocotools.coco import COCO
import json

import matplotlib.pyplot as plt
%matplotlib inline

os.chdir(r"E:\OneDrive\Universidad Autónoma de Bucaramanga\Proyecto de Grado II\Desarrollo\Programa Local")

class CocoSemSegUNET(data.Dataset):
    """
    PyTorch Dataset for semantic segmentation using COCO-style annotations.

    This dataset handles six semantic classes of plant conditions and computes
    class frequencies and weights based on annotated areas for training with 
    class imbalance compensation.

    Attributes:
        coco (COCO): COCO annotation API object.
        image_ids (list): List of image IDs from COCO annotations.
        transform (callable): Transformations to apply to the images.
        split (str): Dataset split ("train" or "test").
        class_names (list): List of class names.
        num_classes (int): Number of classes.
        class_frequencies (np.ndarray): Frequency of each class.
        class_weights (torch.Tensor): Computed weight for each class.
    """
    def __init__(self, coco_json_file, split, transform=None):
        """
        Initialize the dataset.

        Args:
            coco_json_file (str): Path to the COCO annotation JSON file.
            split (str): Dataset split ("train" or "test").
            transform (callable, optional): Image transformation function.
        """
    def _calculate_class_frequencies(self):
        """
        Calculates the pixel area covered by each class across all images.

        Returns:
            np.ndarray: Array of class frequencies (total area per class).
        """
        
    def _calculate_class_weights(self):
        """
        Computes class weights inversely proportional to class area frequency.

        Returns:
            torch.Tensor: Normalized class weights as a tensor.
        """
    def __getitem__(self, index):
        """
        Retrieves the transformed image and multi-channel binary mask for a given index.

        Args:
            index (int): Index of the image to load.

        Returns:
            tuple: Transformed image (Tensor) and masks (Tensor).
        """
    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: Length of the dataset.
        """
    def _generate_masks(self, annotations, image_size):
        """
        Generates binary masks for all defined classes based on annotations.

        Args:
            annotations (list): List of annotations for a single image.
            image_size (tuple): (width, height) of the image.

        Returns:
            torch.Tensor: Multi-channel mask tensor with one channel per class.
        """
    def get_class_weights(self):
        """
        Returns the computed class weights.

        Returns:
            torch.Tensor: Class weights tensor.
        """

def custom_collate_fn(batch):
    """
    Custom collate function to batch images and masks for DataLoader.

    Args:
        batch (list): List of (image, mask) tuples.

    Returns:
        tuple: Batched images and masks.
    """
"""
Dataset and DataLoader setup for training and testing.

- Defines common image transformations.
- Initializes training and testing datasets using COCO-style annotations.
- Creates DataLoaders with a custom collate function.
- Prints class frequencies and computed class weights for both splits.
"""

 ```
⚙️ Funcionalities
    Dataset reading and handling:

    Loading JSON files in COCO format.

    Dataset and DataLoader object construction.

    Generation of batches and minibatches for training.

    Graphical visualization of each image-mask pair.

    Model training:

    Architecture definition (you can specify if you use U-Net, DeepLab, etc.).

    Hyperparameters configuration.

    Training loop with periodic validation.

    Inference and visualization:

    Dedicated script to evaluate new images.

    Graphical interface with tools to load images, run inferences and visualize results with overlay masks.

    


