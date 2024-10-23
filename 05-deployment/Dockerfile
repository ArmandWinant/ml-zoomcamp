FROM python:3.8.12-slim

RUN pip install numpy scikit-learn==0.24.2 flask requests gunicorn

WORKDIR /app

COPY ["predict.py", "model_C=1.0.bin", "./"]