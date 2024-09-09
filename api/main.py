import contextlib
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from models_api.modelsClass import VGG16Model
import shutil
import os
from typing import List, Dict
import pandas as pd

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta

# Secret key pour signer les JWT (à stocker de manière sécurisée, par exemple dans une variable d'environnement)
SECRET_KEY = "votre_cle_secrete_super_securisee"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configuration du hachage de mot de passe (si vous gérez les utilisateurs)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2PasswordBearer indique que l'authentification est basée sur des tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Créer un token JWT
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Vérifier et décoder un token JWT
def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception
    
    @app.post("/token")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        # Pour cet exemple, nous utilisons un utilisateur et un mot de passe codés en dur.
        # Vous pouvez connecter cela à une base de données ou à un service de gestion d'utilisateurs.
        username = form_data.username
        password = form_data.password

        # Exemple utilisateur (vous devez remplacer par un vrai utilisateur provenant de la base de données)
        fake_user = {"username": "user1", "hashed_password": pwd_context.hash("password")}

        # Vérification des identifiants
        if username != fake_user["username"] or not pwd_context.verify(password, fake_user["hashed_password"]):
            raise HTTPException(status_code=400, detail="Incorrect username or password")

        # Créer un token JWT pour l'utilisateur authentifié
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": username}, expires_delta=access_token_expires)

        return {"access_token": access_token, "token_type": "bearer"}



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

class PredictionResponse(BaseModel):
    predicted_class: str

class TrainDataResponse(BaseModel):
    status: str

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
async def predict_image(file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):

    # Vérification du token
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    username = verify_token(token, credentials_exception)



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
    

@app.post("/fine-tune", response_model=TrainDataResponse)
async def fine_tune_model(files: List[UploadFile] = File(...), labels: Dict[str, int] = Form(...)):
    """Fine-tune the model with new data
    \nArgs:
    \n    files (List[UploadFile]): List of image files
    \n    labels (Dict[str, str]): Dictionary of image file names and labels {fileName : labels}"""
    try:
        # Check existence du repertoire temp
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        file_locations = []
        for file in files:
            # Sauvegarde temporaire du fichier
            file_location = os.path.join(temp_dir, file.filename)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_locations.append(file_location)

        # Fine-tune le model
        df = pd.DataFrame(list(labels.items()), columns=['image_file', 'label'])
        df['image_path'] = df.apply(lambda row: os.path.join("temp", row['image_file']), axis=1)

        X_train, X_test, y_train, y_test = train_test_split(df['image_path'], df['label'], test_size=0.33, random_state=42)
        
        dataset_train = model.convert_to_dataset(X_train, y_train)
        dataset_val = model.convert_to_dataset(X_test, y_test)
        
        model.compile_model()
        model.summary()
        
        model.train(train_data=dataset_train, validation_data=dataset_val, epochs=1)
        
        #Get score
        _, test_accuracy = model.evaluate(dataset_val)
        print(f'Test accuracy: {test_accuracy}')

        # Clean up
        for file_location in file_locations:
            os.remove(file_location)

        return TrainDataResponse(status="success")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))