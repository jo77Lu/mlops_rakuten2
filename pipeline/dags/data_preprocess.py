import os
import shutil
import sys
import h5py
from datetime import datetime, timedelta

import pandas as pd
import tensorflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from sklearn.model_selection import train_test_split

# Add models path to sys.path
sys.path.append(os.path.abspath("/app/models"))
from models.modelsClass import VGG16Model

# Constants
IMAGE_SHAPE = (224, 224, 3)
NUM_CLASSES = 27
CSV_PATH = '/app/clean/silverData_vgg16.csv'
TRAINED_MODEL_PATH = "/opt/airflow/pretrain_models"
PRETRAIN_MODEL_FILE = f"{TRAINED_MODEL_PATH}/gold_vgg16.h5"
CANDIDATE_MODEL_FILE = f"{TRAINED_MODEL_PATH}/candidate_vgg16.h5"

# def get_keras_version(model_file):
#     with h5py.File(model_file, 'r') as f:
#         keras_version = f.attrs.get('keras_version')
#         backend = f.attrs.get('backend')
#         return keras_version, backend

def prepare_csv_file(imagesPath, dataFile, labelFile, n_files=None):
    model = VGG16Model(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, include_top=False)
    image_path, labels = model.preprocess_data(imagesPath, labelFile, dataFile, isReduced=True)
    df_vgg16 = pd.DataFrame.from_dict({'image_path': image_path, 'label': labels})
    df_vgg16.to_csv(CSV_PATH, index=False)

def build_and_test_vgg16(pretrain_model_file, trained_model_path, **kwargs):
    """Build and test VGG16 model
    refine a pretrained model with training data

    Args: pretrain_model_file (str): Path to pretrain model file
            trained_model_path (str): Path to save trained model
            **kwargs: Additional arguments        
    """
    # print(f"FILE VERSION: {get_keras_version(pretrain_model_file)}")
    # print(f"KERAS Version: {tensorflow.keras.__version__}")

    ti = kwargs['ti']
    # model = VGG16Model(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, include_top=False)
    model = VGG16Model.from_pretrained(pretrain_model_file)
    model.load_encoder(f"{TRAINED_MODEL_PATH}/encoder.joblib")

    #Download and preprocess train data
    df = pd.read_csv(CSV_PATH)

    X_train, X_test, y_train, y_test = train_test_split(df['image_path'], model.label_encoder.transform(df['label']), test_size=0.33, random_state=42)
    
    dataset_train = model.convert_to_dataset(X_train, y_train)
    dataset_val = model.convert_to_dataset(X_test, y_test)

    print("\n\n#### TESTING FLAG COMPILE START ####\n\n")
    
    model.compile_model()
    model.summary()

    print("\n\n#### TESTING FLAG TRAIN START ####\n\n")
    
    model.train(train_data=dataset_train, validation_data=dataset_val, epochs=1)

    print("\n\n#### TESTING FLAG TRAIN DONE ####\n\n")
    
    #Get score
    _, test_accuracy = model.evaluate(dataset_val)
    print(f'Test accuracy: {test_accuracy}')
    
    #Pushing the accuracy to XCom
    ti.xcom_push(key="candidate_accuracy", value=test_accuracy)
    
    #Saving
    model.save_model(f"{trained_model_path}/candidate_vgg16.h5")

def load_gold_and_test_vgg16(modelFile, **kwargs):
    """Load gold model and test it
    Load the gold model and test it with test data
        Args: modelFile (str): Path to model file
            **kwargs: Additional arguments
    """

    ti = kwargs['ti']

    # model = VGG16Model(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, include_top=False)
    model = VGG16Model.from_pretrained(modelFile)
    model.load_encoder(f"{TRAINED_MODEL_PATH}/encoder.joblib")

    #download and preprocess test data
    df = pd.read_csv(CSV_PATH)
    _, X_test, _, y_test = train_test_split(df['image_path'], model.label_encoder.transform(df['label']), test_size=0.33, random_state=42)
    dataset_val = model.convert_to_dataset(X_test, y_test)

    #Compile and evaluate
    model.compile_model()
    model.summary()

    _, test_accuracy = model.evaluate(dataset_val)
    print(f'Test accuracy: {test_accuracy}')

    #Pushing the accuracy to XCom
    ti.xcom_push(key="gold_accuracy", value=test_accuracy)

def choose_best_model(gold_file, candidate_file, **kwargs):
    """Choose the best model
    Choose the best model between gold and candidate
        Args: gold_file (str): Path to gold model file
            candidate_file (str): Path to candidate model file
            **kwargs: Additional
    """
    ti = kwargs['ti']

    #Get scores
    candidate_score = eval(str(ti.xcom_pull(key="candidate_accuracy", task_ids=['VGG16_build_and_test'])))
    gold_score = eval(str(ti.xcom_pull(key="gold_accuracy", task_ids=['VGG16_Gold_test'])))
    
    #Choose the best model
    if candidate_score > gold_score:
        print("Candidate model is better")
        os.remove(gold_file)
        shutil.move(candidate_file, gold_file)
    else:
        print("GOLD model is better")
        os.remove(candidate_file)


#################################
# DAG

with DAG(
    dag_id='Project_Rakuten2',
    description='My first DAG created with DataScientest',
    tags=['project', 'datascientest', 'Rakuten2'],
    schedule_interval='*/10 * * * *',
    catchup=False,
    default_args={
        'owner': 'airflow',
        'start_date': datetime.now() - timedelta(minutes=1),
    }
) as my_dag:

    my_sensor = FileSensor(
        task_id="check_file",
        fs_conn_id="my_filesystem_connection",
        filepath="/app/rawData/X_train_update.csv",
        poke_interval=30,
        timeout=5 * 30,
        mode='reschedule'
    )

    prepare_file = PythonOperator(
        task_id='Prepare_CSV_File',
        python_callable=prepare_csv_file,
        op_kwargs={
            "imagesPath": "/app/rawData/images/fine_tuning/",
            "dataFile": "/app/rawData/X_train_update.csv",
            "labelFile": "/app/rawData/Y_train_CVw08PX.csv"
        }
    )

    vgg16_build_and_test = PythonOperator(
        task_id='VGG16_build_and_test',
        python_callable=build_and_test_vgg16,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
        op_kwargs={
            "pretrain_model_file": PRETRAIN_MODEL_FILE,
            "trained_model_path": TRAINED_MODEL_PATH
        }
    )

    vgg16_load_gold_and_test = PythonOperator(
        task_id='VGG16_Gold_test',
        python_callable=load_gold_and_test_vgg16,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
        op_kwargs={"modelFile": PRETRAIN_MODEL_FILE}
    )

    choose_best_model = PythonOperator(
        task_id='Choose_best_vgg16',
        python_callable=choose_best_model,
        provide_context=True,
        execution_timeout=timedelta(minutes=3),
        op_kwargs={
            "gold_file": PRETRAIN_MODEL_FILE,
            "candidate_file": CANDIDATE_MODEL_FILE
        }
    )

    # Task dependencies
    my_sensor >> prepare_file >> [vgg16_build_and_test, vgg16_load_gold_and_test]
    [vgg16_build_and_test, vgg16_load_gold_and_test] >> choose_best_model