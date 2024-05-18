from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'email': ['reyhan.merekar@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'boston_ml_pipeline',
    default_args=default_args,
    description='An Airflow DAG to run EDA and train a model',
    schedule_interval=None,
)

# Define tasks
run_eda = BashOperator(
    task_id='run_eda',
    bash_command='python /Users/reymerekar/Desktop/ml_pipeline_airflow/scripts/eda.py',
    dag=dag,
)

train_model = BashOperator(
    task_id='train_model',
    bash_command='python /Users/reymerekar/Desktop/ml_pipeline_airflow/scripts/train.py',
    dag=dag,
)


# Set task dependencies
run_eda >> train_model