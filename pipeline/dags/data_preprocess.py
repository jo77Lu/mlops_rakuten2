import os

import sys
import pandas as pd
from datetime import datetime, timedelta

from airflow import DAG 
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import XCom
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("/app/models"))

from models.modelsClass import VGG16Model


def prepare_csv_file(imagesPath, dataFile, labelFile, n_files=None):
    model = VGG16Model(input_shape=(224, 224, 3), num_classes=27, include_top=False)
    
    image_path, labels = model.preprocess_data(imagesPath,labelFile, dataFile,  isReduced=True)

    df_vgg16 = pd.DataFrame.from_dict({'image_path': image_path, 'label': labels})

    df_vgg16.to_csv(os.path.join('/app/clean', 'silverData_vgg16.csv'), index=False)


def build_and_test_vgg16():
    model = VGG16Model(input_shape=(224, 224, 3), num_classes=27, include_top=False)

    df = pd.read_csv('/app/clean/silverData_vgg16.csv')

    X_train, X_test, y_train, y_test = train_test_split(df['image_path'], df['label'], test_size=0.33, random_state=42)

    print(len(X_train))

    for image_path in X_train:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")

    dataset_train = model.convert_to_dataset(X_train, y_train)
    dataset_val = model.convert_to_dataset(X_test, y_test)

    print(X_train['image_path'].loc[0])

    # Compilez le modèle
    model.compile_model()

    # Affichez le résumé du modèle
    model.summary()

    print(len(os.listdir("/app/rawData/images/image_train/")))
    # Entraînez le modèle
    model.train(train_data=dataset_train, validation_data=dataset_val, epochs=1)

    # Évaluez le modèle
    test_loss, test_accuracy = model.evaluate(dataset_val)
    print(f'Test accuracy: {test_accuracy}')


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

    # Définition de la fonction à exécuter
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
        op_kwargs = {"imagesPath" : "/app/rawData/images/fine_tuning/", "dataFile" : "/app/rawData/X_train_update.csv", "labelFile" : "/app/rawData/Y_train_CVw08PX.csv" }
    )

    vgg16_build_and_test = PythonOperator(
        task_id='VGG16_build_and_test',
        python_callable=build_and_test_vgg16,
        execution_timeout=timedelta(minutes=3),
    )


    # Links:
    my_sensor >> prepare_file >> vgg16_build_and_test
