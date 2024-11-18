# Databricks notebook source
# MAGIC %pip install -q mlflow==2.16.2 databricks-vectorsearch==0.40 databricks-sdk==0.28.0 langchain==0.3.0 langchain-community==0.3.0 mlflow[databricks] databricks-agents==0.6.0 

# COMMAND ----------

import pytz
from datetime import datetime
#TODO:
####CHANGE ME

#Timezone that you want to use for mlflow logging
timezone_for_logging = "US/Eastern"
logging_timezone = pytz.timezone(timezone_for_logging)

#catalog to use for creating data tables and keeping other resources
#You need necessary privileges to create, delete tables, functions, and models
catalog = "main"

#schema to use for creating data tables and keeping other resources
#You need necessary privileges to create, delete tables, functions, and model
schema = "covid_trials"

#The Volume folder where data file will be copied to
data_folder = "data"

#the covid data file
covid_data_file_name = "COVID clinical trials.csv"

#covid cleaned data table name
covid_data_table_name = "covid_data"

#MLflow experiment tag
experiment_tag = f"covid_trials"

# COMMAND ----------

current_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
project_root_path = "/".join(current_path.split("/")[1:-2])

# COMMAND ----------

db_host_name = spark.conf.get('spark.databricks.workspaceUrl')
db_host_url = f"https://{db_host_name}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
user_name = user_email.split('@')[0].replace('.','_')
user_prefix = f"{user_name[0:4]}{str(len(user_name)).rjust(3, '0')}"

# COMMAND ----------

#Create mlflow experiment
import mlflow
from databricks.sdk import WorkspaceClient

mlflow_experiment_base_path = f"Users/{user_email}/mlflow_experiments"

def set_mlflow_experiment(experiment_tag):
    w = WorkspaceClient()
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}_{user_prefix}"
    return mlflow.set_experiment(experiment_path)

# COMMAND ----------

print(f"Using catalog: {catalog}")
print(f"Using schema: {schema}")
print(f"Project root: {project_root_path}")
print(f"MLflow Experiment Path: {mlflow_experiment_base_path}")
