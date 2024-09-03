from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from models_api.modelsClass import VGG16Model
import shutil
import os


CURRENT_DIR = os.path.dirname(__file__)
GOLD_MODEL_FILE = "pretrain_models/gold_vgg16.h5"
GOLD_LABELENCODER_FILE = "pretrain_models/encoder.joblib"
GOLD_MODEL_PATH = os.path.join(CURRENT_DIR, GOLD_MODEL_FILE)
GOLD_LABELENCODER_PATH = os.path.join(CURRENT_DIR, GOLD_LABELENCODER_FILE)

app = FastAPI()

# Initialize the model
model = VGG16Model.from_pretrained(GOLD_MODEL_PATH)
model.load_encoder(GOLD_LABELENCODER_PATH)


# @app.post("/predict", response_model=PredictionResponse)
# async def predict_image(file: UploadFile = File(...)):
#     try:
#         # Save the uploaded file
#         file_location = f"temp/{file.filename}"
#         with open(file_location, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         # Make prediction
#         prediction = model.predict(file_location)
#         pred_class= model.predict_class(file_location)

#         # Clean up the saved file
#         os.remove(file_location)

#         return pred_class
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/model-summary")
async def get_model_summary():
    try:
        summary = model.summary()
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))