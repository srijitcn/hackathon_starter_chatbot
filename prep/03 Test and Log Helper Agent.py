# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

import os

os.environ["BRAVE_API_KEY"] = dbutils.secrets.get("multi_agent","web_search_api_key")

from agents.helper_agent import helper_chain, helper_chain_config

# COMMAND ----------

response = helper_chain.invoke({"messages":[{"content": "whats the capital of france" , "role": "user"}] })

print(response)

# COMMAND ----------

response = helper_chain.invoke({"messages":[{"content": "How many florida can fit in India?" , "role": "user"}] })

# COMMAND ----------

set_mlflow_experiment("covid19_agent")

## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=helper_chain)

with mlflow.start_run(run_name="helper_agent"):
  logged_chain_info = mlflow.langchain.log_model(
          #Note: In classical ML, MLflow works by serializing the model object.  In generative AI, chains often include Python packages that do not serialize.  Here, we use MLflow's new code-based logging, where we saved our chain under the chain notebook and will use this code instead of trying to serialize the object.
          lc_model=os.path.join(os.getcwd(), "agents/helper_agent.py"),  # Chain code file  
          model_config="config/helper_agent_config.yaml", 
          artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
          input_example=helper_chain_config.get("input_example"),
          example_no_conversion=True  # Required by MLflow to use the input_example as the chain's schema          
      )
