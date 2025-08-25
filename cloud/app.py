import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from cloud.inference import CannabisSegmentationModel
from PIL import Image, ImageDraw
import io
import numpy as np
import uuid

# Initialize

app = FastAPI(title='Cannabis Disease Segmentation App')

static_dir = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(static_dir, exist_ok=True)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), 'templates'))

modelpath = os.path.join(os.path.dirname(__file__), '..', 'model_Unet__46_checkpoint_epoch_40_v2.pt')
model = CannabisSegmentationModel(modelpath)

class_names = [
    'Non-plant area',
    'Healthy Plants',
    'Early Botrytis (Stage 1)',
    'Advanced Botrytis (Stage 2)',
    'Non-reverse Botrytis (Stage 3)',
    'Nutritional Deficiencies',
]

class_colors = {
    0: (0, 0, 0, 0),            # Background
    1: (0, 255, 0, 255),        # Healthy Plants
    2: (165, 42, 42, 255),      # Early Botrytis (Stage 1)
    3: (128, 0, 128, 255),      # Advanced Botrytis (Stage 2)
    4: (255, 165, 0, 255),      # Non-reverse Botrytis (Stage 3)
    5: (255, 255, 0, 255),      # Nutritional Deficiencies
}

def create_mask_image(mask_np):
    h, w = mask_np.shape
    mask_img = Image.new('RGBA', (w, h))

    for class_id, color in class_colors.items():
        mask_class = (mask_np == class_id)
        if np.any(mask_class):
            draw = ImageDraw.Draw(mask_img)
            draw.bitmap(
                (0,0),
                Image.fromarray((mask_class*255).astype('uint8'), mode='L'),
                fill=color
            )
    

    return mask_img

def showRecommendations(class_percentages):
    """Show specific recommendations based on the percentages found"""
    recommendation_text = "Recommendations:\n"

    thresholds_and_recommendations = {
        'Early Botrytis (Stage 1)': {
            'threshold': 1.5,
            'text': "* Early Botrytis above the threshold detected. Inspect the affected areas and apply preventive fungicides."
        },
        'Advanced Botrytis (Stage 2)': {
            'threshold': 1.5,
            'text': "* Advanced Botrytis above the detected threshold. Remove affected plants and apply more powerful fungicides."
        },
        'Non-reverse Botrytis (Stage 3)': {
            'threshold': 1.5,
            'text': "* Non-reverse Botrytis above the threshold detected. Remove severely affected plants to prevent spread."
        },
        'Nutritional Deficiencies': {
            'threshold': 1.5,
            'text': "* Nutritional deficiencies above the threshold detected. Perform a soil analysis and adjust the fertilizers."
        }
    }

    for class_name, info in thresholds_and_recommendations.items():
        if class_percentages.get(class_name, 0) >= info['threshold']:
            recommendation_text += info['text'] + "\n"

    if recommendation_text.strip() == "Recommendations:":
        recommendation_text += "\n* No critical issues detected."

    return recommendation_text

app.mount('/static', StaticFiles(directory=static_dir), name='static')

@app.get('/')
async def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request, 'result': None})

@app.post('/predict')
async def predict(
    request: Request,
    file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_resized = image.resize((1024, 1024))

        # Prediction
        mask_np = model.predict_mask(image_resized)
        class_pixels, class_percentages = model.calculate_class_metrics(mask_np)

        mask_img = create_mask_image(mask_np)

        # Temporal saving

        orig_filename = f'original_{uuid.uuid4().hex}.png'
        orig_path = os.path.join(static_dir, orig_filename)
        image_resized.save(orig_path)

        mask_filename = f'mask_{uuid.uuid4().hex}.png'
        mask_path = os.path.join(static_dir, mask_filename)
        mask_img.save(mask_path)

        pixels_dictionary = {class_names[i]: int(v) for i, v in class_pixels.items()}
        percentages_dict = {class_names[i]: float(f'{v:.2f}') for i, v in class_percentages.items()}

        recommendations = showRecommendations(percentages_dict)
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": {
                    "original_url": f"/static/{orig_filename}",
                    "mask_url": f"/static/{mask_filename}",
                    "class_pixels": pixels_dictionary,
                    "class_percentages": percentages_dict,
                    "recommendations": recommendations,
                },
            },
        )
    
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", {"request": request, "result": {"error": str(e)}}
        )
