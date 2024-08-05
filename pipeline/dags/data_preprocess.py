import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta

# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from joblib import dump

from airflow import DAG 
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import XCom


rawPath="/app/raw_files"


def prepare_csv_file(imagesPath, filename, n_files=None):
    
    dfs = []


    with open(filename, 'r') as file:
        data_temp = json.load(file)
        print(type(data_temp))
        print(data_temp)
    if isinstance(data_temp,list):
        for data_city in data_temp:
            print(type(data_city))
            print(data_city)
            dfs.append(
                {
                    'temperature': data_city['main']['temp'],
                    'city': data_city['name'],
                    'pression': data_city['main']['pressure'],
                    
                }
            )
    else:
        dfs.append(
            {   
                'temperature': data_temp['main']['temp'],
                'city': data_temp['name'],
                'pression': data_temp['main']['pressure'],
                
            }   
        ) 

    df = pd.DataFrame(dfs)

    print('\n', df.head(10))

    df.to_csv(os.path.join('/app/clean_data', filename), index=False)


# # MACHINE LEARNING:
# def compute_model_score(model,path_to_data='/app/clean_data/fulldata.csv', **kwargs):
#     X, y = prepare_data(path_to_data)
#     ti = kwargs['ti']
#     keyName = type(model).__name__.split(".")[-1]
#     # computing cross val
#     cross_validation = cross_val_score(
#         model,
#         X,
#         y,
#         cv=3,
#         scoring='neg_mean_squared_error')

#     model_score = cross_validation.mean()
#     ti.xcom_push(key=keyName, value=model_score)
#     return model_score


# def train_and_save_model(model, path_to_data='../clean_data/fulldata.csv', path_to_model='./app/model.pckl'):
#     X, y = prepare_data(path_to_data)
#     # training the model
#     model.fit(X, y)
#     os.makedirs(os.path.dirname(path_to_model), exist_ok=True)
#     # saving model
#     print(str(model), 'saved at ', path_to_model)
#     dump(model, path_to_model)


# def prepare_data(path_to_data='../clean_data/fulldata.csv'):

    
#     # reading data
#     df = pd.read_csv(path_to_data)
    
#     # ordering data according to city and date
#     df = df.sort_values(['city', 'date'], ascending=True)
    
#     dfs = []

#     for c in df['city'].unique():
#         df_temp = df[df['city'] == c]
#         print("FLAG: ")
#         print(df_temp.head())
#         # creating target
#         df_temp.loc[:, 'target'] = df_temp['temperature'].shift(1)

#         # creating features
#         for i in range(1, 10):
#             df_temp.loc[:, 'temp_m-{}'.format(i)
#                         ] = df_temp['temperature'].shift(-i)

#         # deleting null values
#         df_temp = df_temp.dropna()

#         dfs.append(df_temp)

#     # concatenating datasets
#     df_final = pd.concat(
#         dfs,
#         axis=0,
#         ignore_index=False
#     )

#     # deleting date variable
#     df_final = df_final.drop(['date'], axis=1)

#     # creating dummies for city variable
#     df_final = pd.get_dummies(df_final)

#     print(df_final.head())

#     features = df_final.drop(['target'], axis=1)
#     target = df_final['target']

#     return features, target 


# def choose_train_and_save(path_to_data='/app/clean_data/fulldata.csv', path_to_model='./app/model.pckl',**kwargs):
#     ti=kwargs['ti']
#     score =[]

#     score.append(eval(str(ti.xcom_pull(key="LinearRegression", task_ids=["Score_Linear_Regression"])))[0])
#     score.append(eval(str(ti.xcom_pull(key="DecisionTreeRegressor", task_ids=["Score_Decision_Tree"])))[0])
#     score.append(eval(str(ti.xcom_pull(key="RandomForestRegressor", task_ids=["Score_Random_Forest"])))[0])

#     print(score)

#     index_of_max = score.index(max(score))

#     if index_of_max == 0:
#         train_and_save_model(LinearRegression(), path_to_data=path_to_data, path_to_model=path_to_model)
#     elif index_of_max == 1:
#         train_and_save_model(DecisionTreeRegressor(), path_to_data=path_to_data, path_to_model=path_to_model)
#     elif index_of_max == 2:
#         train_and_save_model(RandomForestRegressor(), path_to_data=path_to_data, path_to_model=path_to_model)
#     else:
#         raise ValueError("Impossible de selectionner un modele")



#################################
# DAG

with DAG(
    dag_id='Project_Rakuten2',
    description='My first DAG created with DataScientest',
    tags=['project', 'datascientest', 'Rakuten2'],
    schedule_interval='*/1 * * * *',
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
        op_kwargs = {"imagesPath" : "app/rawData/images/image_train", "dataFile" : "app/rawData/X_train_update.csv", "labelFile" : "Y_train_CVw08PX.csv" }
    )

    # transform_top_20 = PythonOperator(
    #     task_id='Transform_top_20',
    #     python_callable= transform_data_into_csv,
    #     op_kwargs = {"n_files" : 20}
    # )

    # transform_all = PythonOperator(
    #     task_id='Transform_all',
    #     python_callable= transform_data_into_csv,
    #     op_kwargs = {"filename" : "fulldata.csv"}
    # )

    # compute_score_LinReg = PythonOperator(
    #     task_id='Score_Linear_Regression',
    #     provide_context=True,
    #     python_callable= compute_model_score,
    #     op_kwargs = {"model" : LinearRegression()}
    # )
    # compute_score_DecTree = PythonOperator(
    #     task_id='Score_Decision_Tree',
    #     provide_context=True,
    #     python_callable= compute_model_score,
    #     op_kwargs = {"model" : DecisionTreeRegressor()}
    # )
    # compute_score_RandFrst = PythonOperator(
    #     task_id='Score_Random_Forest',
    #     provide_context=True,
    #     python_callable= compute_model_score,
    #     op_kwargs = {"model" : RandomForestRegressor()}
    # )

    # select_and_train = PythonOperator(
    #     task_id='Select_and_Train',
    #     provide_context=True,
    #     python_callable=choose_train_and_save,
    # )

    # Links:
    my_sensor >> prepare_file
    # get_raw_data >> [transform_top_20, transform_all]
    # transform_all >> [compute_score_LinReg, compute_score_DecTree, compute_score_RandFrst]
    # compute_score_LinReg >> select_and_train
    # compute_score_DecTree >> select_and_train
    # compute_score_RandFrst >> select_and_train