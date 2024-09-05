import contextlib
import io
import os
import shutil
from fastapi import Body, FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from models_api.modelsClass import VGG16Model
from typing import List, Dict
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_MODEL_FILE = "pretrain_models/gold_vgg16.h5"
GOLD_LABELENCODER_FILE = "pretrain_models/encoder.joblib"
GOLD_MODEL_PATH = os.path.join(CURRENT_DIR, GOLD_MODEL_FILE).replace("\\", "/")
GOLD_LABELENCODER_PATH = os.path.join(CURRENT_DIR, GOLD_LABELENCODER_FILE).replace("\\", "/")

# Verify the existence of model and encoder files
if not os.path.exists(GOLD_MODEL_PATH):
    print(f"Error: Model file does not exist at {GOLD_MODEL_PATH}")
if not os.path.exists(GOLD_LABELENCODER_PATH):
    print(f"Error: Encoder file does not exist at {GOLD_LABELENCODER_PATH}")

app = FastAPI()

class PredictionResponse(BaseModel):
    predicted_class: str

class TrainDataResponse(BaseModel):
    status: str
    history: Dict[str, List[float]]
    loss: float
    accuracy: float

class EvalDataResponse(BaseModel):
    status: str
    loss: float
    accuracy: float

class LabelData(BaseModel):
    labels: Dict[str, int]

def convert_to_dict(input_str: str) -> Dict[str, int]:
    key, value = input_str.split(':')
    return {key.strip(): int(value.strip())}

def read_csv(file: UploadFile) -> pd.DataFrame:
    try:
        df = pd.read_csv(file.file)
        if 'filePath' not in df.columns or 'labels' not in df.columns:
            raise ValueError("CSV file must contain 'filePath' and 'labels' columns")
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def save_temp_files(file_paths: List[str], temp_dir: str) -> List[str]:
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_locations = []
    for file_path in file_paths:
        file_location = os.path.join(temp_dir, os.path.basename(file_path))
        with open(file_path, "rb") as f:
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(f, buffer)
        file_locations.append(file_location)
    return file_locations

def clean_up_files(file_locations: List[str]):
    for file_location in file_locations:
        os.remove(file_location)

# Initialization du model
try:
    model = VGG16Model.from_pretrained(GOLD_MODEL_PATH)
    model.load_encoder(GOLD_LABELENCODER_PATH)
except Exception as e:
    print(f"Error loading model or encoder: {e}")
    raise

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/model-summary")
async def get_model_summary():
    try:
        # Capture the model summary
        summary_io = io.StringIO()
        with contextlib.redirect_stdout(summary_io):
            model.summary()
        summary_str = summary_io.getvalue()
        return {"summary": summary_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        # Check existence du repertoire temp
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Sauvegarde temporaire du fichier
        file_location = os.path.join(temp_dir, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pred_class = model.predict_class(file_location)

        # Clean up
        os.remove(file_location)

        return PredictionResponse(predicted_class=str(pred_class))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fine-tune", response_model=TrainDataResponse)
async def fine_tune_model(csv_file: UploadFile = File(...), test_size: float = Form(0.33), epochs: int = Form(5)):
    """Fine-tune the model with new data
    \nArgs:
    \n    csv_file (UploadFile): CSV file with columns 'filePath' and 'labels'"""

    try:
        # Read the CSV file
        df = read_csv(csv_file)
        
        # Check existence of temp directory
        temp_dir = "temp"
        file_locations = save_temp_files(df["filePath"].tolist(), temp_dir)

        # Fine-tune the model
        df['image_path'] = df['filePath'].apply(lambda x: os.path.join(temp_dir, os.path.basename(x)))
        X_train, X_test, y_train, y_test = train_test_split(df['image_path'], df['labels'], test_size=test_size, random_state=42)
        
        dataset_train = model.convert_to_dataset(X_train, y_train)
        dataset_val = model.convert_to_dataset(X_test, y_test)
        
        model.compile_model()
        history = model.train(train_data=dataset_train, validation_data=dataset_val, epochs=epochs)
        
        # Get score
        loss, accuracy = model.evaluate(dataset_val)

        # Clean up
        clean_up_files(file_locations)

        return TrainDataResponse(status="success", history=history.history, loss=loss, accuracy=accuracy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvalDataResponse)
async def evaluate_model(csv_file: UploadFile = File(...)):
    """Evaluate the model with new data
    \nArgs:
    \n    csv_file (UploadFile): CSV file with columns 'filePath' and 'labels'"""

    try:
        # Read the CSV file
        df = read_csv(csv_file)
        
        # Check existence of temp directory
        temp_dir = "temp"
        file_locations = save_temp_files(df["filePath"].tolist(), temp_dir)

        # Evaluate the model
        df['image_path'] = df['filePath'].apply(lambda x: os.path.join(temp_dir, os.path.basename(x)))
        dataset_eval = model.convert_to_dataset(df['image_path'], df['labels'])
        
        model.compile_model()
        loss, accuracy = model.evaluate(dataset_eval)

        # Clean up
        clean_up_files(file_locations)

        return EvalDataResponse(status="success", loss=loss, accuracy=accuracy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))