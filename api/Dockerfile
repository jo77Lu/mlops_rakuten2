# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

COPY ./requirements_api.txt /app/requirements_api.txt

# Grant execution rights
RUN chmod +x ./wait_for_it.sh

# Copy the current directory contents into the container at /app
COPY ./main.py /app/main.py
COPY ./wait_for_it.sh /app/wait_for_it.sh
COPY ./pretrain_models /app/pretrain_models
COPY ./models_api /app/models_api
COPY ./testData /app/testData

# Update pip
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_api.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]