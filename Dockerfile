#   Pytorch image optimized for CPU
FROM pytorch/pytorch:latest

# Avoid .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

#   Main container directory
WORKDIR /app

#   Copy requirements
COPY cloud/requirements.txt .

#   Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#   Copy Source code
COPY cloud/ ./cloud

#   Copy Model Weights
COPY model_Unet__46_checkpoint_epoch_40_v2.pt ./model_Unet__46_checkpoint_epoch_40_v2.pt

# Expose port
EXPOSE 8080

#   Start server with availability to dynamic port
CMD ["sh", "-c", "uvicorn cloud.app:app --host 0.0.0.0 --port ${PORT:-8080}"]