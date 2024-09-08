import os
import pytest
import requests
import sys
from io import BytesIO
from fastapi.testclient import TestClient

# URL de base de l'application FastAPI
RUNNING_IN_DOCKER = os.getenv("RUNNING_IN_DOCKER", "false").lower() == "true"

if RUNNING_IN_DOCKER:
    from main import app
    client = TestClient(app)
else:
    BASE_URL = "http://localhost:8080"


def test_health_check():
    """Test the /health endpoint"""
    if RUNNING_IN_DOCKER:
        response = client.get("/health")
    else:
        response = requests.get(f"{BASE_URL}/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_model_summary():
    """Test the /model-summary endpoint"""
    if RUNNING_IN_DOCKER:
        response = client.get("/model-summary")
    else:
        response = requests.get(f"{BASE_URL}/model-summary")
    
    assert response.status_code == 200
    assert "summary" in response.json()

def test_predict_success():
    """Test the /predict endpoint with a valid image"""
    file_path = os.path.join(os.getcwd(), "tests", "app_basis", "data", "mock_image.jpg")
    with open(file_path, "rb") as file:
        files = {"file": ("mock_image.jpg", file, "image/jpeg")}
        if RUNNING_IN_DOCKER:
            response = client.post("/predict", files=files)
        else:
            response = requests.post(f"{BASE_URL}/predict", files=files)
    
    assert response.status_code == 200
    assert "predicted_class" in response.json()

def test_predict_invalid_file_extension():
    """Test the /predict endpoint with a non-image file (invalid extension)"""
    file_path = os.path.join(os.getcwd(), "tests", "app_basis", "data", "mock_text.txt")
    with open(file_path, "rb") as file:
        files = {"file": ("mock_text.txt", file, "text/plain")}
        if RUNNING_IN_DOCKER:
            response = client.post("/predict", files=files)
        else:
            response = requests.post(f"{BASE_URL}/predict", files=files)
    assert response.status_code == 500  # Unprocessable entity

def test_fine_tune_missing_file():
    """Test the /fine-tune endpoint with no file uploaded"""
    data = {'test_size': '0.2', 'epochs': '3'}
    if RUNNING_IN_DOCKER:
        response = client.post("/fine-tune", data=data)
    else:
        response = requests.post(f"{BASE_URL}/fine-tune", data=data)
    print(response.status_code)
    assert response.status_code == 422  # ,Missing file


def test_fine_tune_invalid_epochs():
    """Test the /fine-tune endpoint with invalid epochs (negative)"""
    file_path = os.path.join(os.getcwd(), "tests", "app_basis", "data", "mock_fine_tune.csv")
    with open(file_path, "rb") as file:
        files = {'csv_file': ('mock_fine_tune.csv', file, 'text/csv')}
        data = {'test_size': '0.2', 'epochs': '-1'}
        if RUNNING_IN_DOCKER:
            response = client.post("/fine-tune", files=files, data=data)
        else:
            response = requests.post(f"{BASE_URL}/fine-tune", files=files, data=data)
    print(response.status_code)
    assert response.status_code == 500
    

# Test nominal : Fichier CSV valide
def test_evaluate_success():
    file_path = os.path.join(os.getcwd(), "testData", "testData.csv")
    with open(file_path, "rb") as file:
        files = {'csv_file': ('mock_fine_tune.csv', file, 'text/csv')}
        if RUNNING_IN_DOCKER:
            response = client.post("/evaluate", files=files)
        else:
            response = requests.post(f"{BASE_URL}/evaluate", files=files)
    
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    assert "loss" in json_response
    assert "accuracy" in json_response

# Test limite : Fichier CSV vide
def test_evaluate_empty_csv():
    empty_csv = BytesIO(b"")
    files = {'csv_file': ('empty.csv', empty_csv, 'text/csv')}
    if RUNNING_IN_DOCKER:
        response = client.post("/evaluate", files=files)
    else:
        response = requests.post(f"{BASE_URL}/evaluate", files=files)
    
    assert response.status_code == 400
    assert response.json() == {"detail" : "400: No columns to parse from file"} 

def test_evaluate_missing_filePath():
    file_path = os.path.join(os.getcwd(), "testData", "mock_evaluate_missing_filepath.csv")
    with open(file_path, "rb") as file:
        files = {'csv_file': ('mock_evaluate_missing_filepath.csv', file, 'text/csv')}
        if RUNNING_IN_DOCKER:
            response = client.post("/evaluate", files=files)
        else:
            response = requests.post(f"{BASE_URL}/evaluate", files=files)
    
    assert response.status_code == 400
    assert response.json() == {"detail" : "400: CSV file must contain 'filePath' and 'labels' columns"} 

def test_evaluate_missing_filePath():
    file_path = os.path.join(os.getcwd(), "testData", "mock_evaluate_missing_labels.csv")
    with open(file_path, "rb") as file:
        files = {'csv_file': ('mock_evaluate_missing_labels.csv', file, 'text/csv')}
        if RUNNING_IN_DOCKER:
            response = client.post("/evaluate", files=files)
        else:
            response = requests.post(f"{BASE_URL}/evaluate", files=files)
    
    assert response.status_code == 400
    assert response.json() == {"detail" : "400: CSV file must contain 'filePath' and 'labels' columns"} 


def test_evaluate_non_csv_file():
    non_csv_file = os.path.join(os.getcwd(), "testData", "mock_text.txt")
    with open(non_csv_file, "rb") as file:
        files = {'csv_file': ('mock_text.txt', file, 'text/csv')}
        if RUNNING_IN_DOCKER:
            response = client.post("/evaluate", files=files)
        else:
            response = requests.post(f"{BASE_URL}/evaluate", files=files)
    
    assert response.status_code == 500 


def test_evaluate_invalid_file_paths():
    invalid_csv_content = "filePath,labels\npath/to/nonexistent1.jpg,cat\npath/to/nonexistent2.jpg,dog"
    invalid_csv = BytesIO(invalid_csv_content.encode('utf-8'))
    files = {'csv_file': ('invalid_paths.csv', invalid_csv, 'text/csv')}
    if RUNNING_IN_DOCKER:
        response = client.post("/evaluate", files=files)
    else:
        response = requests.post(f"{BASE_URL}/evaluate", files=files)
    
    assert response.status_code == 500
