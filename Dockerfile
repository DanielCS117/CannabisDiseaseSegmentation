FROM python:3.11-slim

WORKDIR /app

COPY cloud/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY cloud/ ./cloud

COPY model_Unet__46_checkpoint_epoch_40_v2.pt ./model_Unet__46_checkpoint_epoch_40_v2.pt

EXPOSE 5000

CMD ["sh", "-c", "uvicorn cloud.app:app --host 0.0.0.0 --port ${PORT:-8080}"]