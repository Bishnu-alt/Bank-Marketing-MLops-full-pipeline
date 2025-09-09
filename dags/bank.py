from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os
from airflow.providers.mysql.operators.mysql import MySqlOperator

# Connection and DB details
MYSQL_CONN_ID = "my_mysql" 
DB_NAME = "bank"
TABLE_NAME = "bank_data"
project_folder = "/home/bishnu-upadhyay/projects/MLops"
sys.path.append(project_folder)

# Import Python scripts
import schema_table
import data_injection
import preprocessing
import model_train
import model_deploy
import ge_validation
import datadrift
import conceptdrift

# Default args with email on failure
default_args = {
    "owner": "bishnu",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email": ["bishnuprasadupadhyay3@gmail.com"],   
    "email_on_failure": False,              
}

with DAG(
    dag_id="bank_ml_full_pipeline_dag",
    default_args=default_args,
    description="ML pipeline DAG with environment setup: Database schema -> Data Injection -> Preprocessing -> Training -> Deployment",
    schedule="@daily",
    start_date=datetime(2025, 8, 25),
    catchup=False,
    tags=["ml", "pipeline", "full"],
) as dag:

    # ===== Bash Operators for environment setup =====
    activate_env = BashOperator(
        task_id='activate_env',
        bash_command='conda run -n mlopsenv /bin/bash -c "echo Environment activated"'
    )

    start_docker = BashOperator(
        task_id='start_docker',
        bash_command='docker start mariadb_container redis_container'
    )

    docker_exec = BashOperator(
        task_id='docker_exec',
        bash_command='docker exec -i mariadb_container mariadb -u bishnu -p"bishnu@pass123" bank -e "SELECT 1;"'
    )
    start_mlflow = BashOperator(
    task_id="mlflow_task",
    bash_command=(
        "nohup conda run -n mlopsenv mlflow ui "
        "--host 0.0.0.0 --port 5000 "
        "--backend-store-uri file:///home/bishnu-upadhyay/projects/MLops/mlflow/mlruns "
        "--default-artifact-root /home/bishnu-upadhyay/projects/MLops/mlflow/artifacts "
        "> ~/mlflow.log 2>&1 & "
        "echo $! > /tmp/mlflow_ui_pid.txt"
    )
    )
    # ===== Python Operators for ML pipeline =====
    create_bigtable = PythonOperator(
        task_id="create_bank_bigtable",
        python_callable=schema_table.create_bigtable,
    )
    
    ge_validate = PythonOperator(
        task_id="ge_validate_bank_data",
        python_callable=ge_validation.run_phishing_ge_validation,
        op_kwargs={"bank_csv_path": "/home/bishnu-upadhyay/projects/MLops/data/bank-additional-full.csv"},
    )
    
    data_inject = PythonOperator(
        task_id="data_injection",
        python_callable=data_injection.run,
    )

    preprocess = PythonOperator(
        task_id="data_preprocessing",
        python_callable=preprocessing.run,
        op_kwargs={
            "db_url": "mysql+pymysql://bishnu:bishnu%40pass123@localhost:3306/bank",
            "table_name": "bank_data",
            "train_arrow_file": "/home/bishnu-upadhyay/projects/MLops/data/bank_train_data.parquet",
            "train_output_table": "bank_train_data",
            "output_csv": "/home/bishnu-upadhyay/projects/MLops/data/bank_train_data.csv",
            "test_arrow_file": "/home/bishnu-upadhyay/projects/MLops/data/bank_test_data.parquet",
            "test_output_table": "bank_test_data",
            "test_output_csv": "/home/bishnu-upadhyay/projects/MLops/data/bank_test_data.csv"
        },
    )

    train_model = PythonOperator(
    task_id="model_training",
    python_callable=model_train.run,
    op_kwargs={
        "train_arrow_file": "/home/bishnu-upadhyay/projects/MLops/data/bank_train_data.parquet",
        "test_arrow_file": "/home/bishnu-upadhyay/projects/MLops/data/bank_test_data.parquet",
        "model_path": "/home/bishnu-upadhyay/projects/MLops/model/bank_model.pkl",
        "registered_model_name": "Bank_XGB_Model",
        "model_alias": "Production",
    },
    )

    monitor_drift = PythonOperator(
        task_id="monitor_data_drift_task",
        python_callable=datadrift.monitor_data_drift,
    )
    
    monitor_concept = PythonOperator(
        task_id="monitor_concept_drift_task",
        python_callable=conceptdrift.monitor_concept_drift,
    )

    # Environment setup first
    activate_env >> start_docker >> docker_exec >> start_mlflow >> create_bigtable >> ge_validate >> data_inject >> preprocess >> train_model  >> monitor_drift >> monitor_concept
