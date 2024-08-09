<<<<<<< Updated upstream
#docker-compose
=======
>>>>>>> Stashed changes
version: '3.8'

services:
  api:
    build: ./api
    container_name: ml_api
    ports:
      - "5000:5000"
    depends_on:
      - model
      - db
    environment:
      - MODEL_URL=http://model:8501/v1/models/model
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=dbname
      - DB_USER=user
      - DB_PASSWORD=password

  model:
    image: tensorflow/serving:latest
    container_name: ml_model
    ports:
      - "8501:8501"
    volumes:
      - ./model:/models/model
    environment:
      - MODEL_NAME=model

  db:
    image: postgres:latest
    container_name: ml_db
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: dbname
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
