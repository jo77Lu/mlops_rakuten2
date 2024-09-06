FROM python:3.9-slim

WORKDIR /app

COPY requirements_api.txt .

ENV RUNNING_IN_DOCKER=true

RUN pip install --no-cache-dir -r requirements_api.txt

COPY main.py /app/main.py
COPY pretrain_models /app/pretrain_models
COPY models_api /app/models_api

COPY tests/app_basis/test_app_basis.py /app/tests/
COPY tests/app_basis/data/mock_fine_tune.csv /app/tests/data/
COPY tests/app_basis/data/mock_image.jpg /app/tests/data/
COPY tests/app_basis/data/mock_text.txt /app/tests/data/


CMD ["pytest", "tests/test_app_basis.py"]