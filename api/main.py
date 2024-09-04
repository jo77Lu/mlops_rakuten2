import contextlib
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from models_api.modelsClass import VGG16Model
import shutil
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_MODEL_FILE = "pretrain_models/gold_vgg16.h5"
GOLD_LABELENCODER_FILE = "pretrain_models/encoder.joblib"
GOLD_MODEL_PATH = os.path.join(CURRENT_DIR, GOLD_MODEL_FILE)
GOLD_LABELENCODER_PATH = os.path.join(CURRENT_DIR, GOLD_LABELENCODER_FILE)

# Converti les backslashes en slashes
GOLD_MODEL_PATH = GOLD_MODEL_PATH.replace("\\", "/")
GOLD_LABELENCODER_PATH = GOLD_LABELENCODER_PATH.replace("\\", "/")


# Verifie lexistence des fichiers
if not os.path.exists(GOLD_MODEL_PATH):
    print(f"Error: Model file does not exist at {GOLD_MODEL_PATH}")
if not os.path.exists(GOLD_LABELENCODER_PATH):
    print(f"Error: Encoder file does not exist at {GOLD_LABELENCODER_PATH}")

app = FastAPI()


# Initialization du model
try:
    model = VGG16Model.from_pretrained(GOLD_MODEL_PATH)
    model.load_encoder(GOLD_LABELENCODER_PATH)
except Exception as e:
    print(f"Error loading model or encoder: {e}")
    raise

class PredictionResponse(BaseModel):
    predicted_class: str

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

        pred_class= model.predict_class(file_location)

        # Clean up
        os.remove(file_location)

        return  PredictionResponse(predicted_class=str(pred_class))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))