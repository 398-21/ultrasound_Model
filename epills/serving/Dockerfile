# The hosted API image. Build context = the epills folder, so the paths below match the repository:
#   docker build -f serving/Dockerfile -t pharmascan-api .
#   docker run -p 7860:7860 pharmascan-api
# Only serving/ and src/epills/core are copied in; nothing from training, scripts, data or outputs.
# Hugging Face Spaces (Docker SDK) expects port 7860 and a non-root user with uid 1000.
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY serving/requirements.txt serving/requirements.txt
# CPU-only PyTorch first (~200 MB instead of the CUDA build's ~2 GB); the requirements then find it installed
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r serving/requirements.txt

COPY src/epills/__init__.py src/epills/__init__.py
COPY src/epills/core src/epills/core
COPY serving/app serving/app

ENV PYTHONPATH=/home/user/app/src:/home/user/app \
    HF_HOME=/home/user/.cache/huggingface \
    EPILLS_PRELOAD=1
USER user
EXPOSE 7860
CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
