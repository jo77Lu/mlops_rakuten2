FROM python:3.8-slim

WORKDIR /app

COPY ./requirements.txt ./requirements.txt

COPY ./app/main.py ./main.py

RUN apt-get update && apt-get install -y default-libmysqlclient-dev \
    build-essential \
    pkg-config

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN rm requirements.txt
