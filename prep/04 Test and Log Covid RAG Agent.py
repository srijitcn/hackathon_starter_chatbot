# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

import os
os.environ["VECTOR_SEARCH_PAT"] = dbutils.secrets.get("multi_agent","pat")
#os.environ["VECTOR_SEARCH_CLIENT_ID"] = dbutils.secrets.get("multi_agent","vector_search_client_id")
#os.environ["VECTOR_SEARCH_CLIENT_SECRET"] = dbutils.secrets.get("multi_agent","vector_search_client_secret")
os.environ["WORKSPACE_URL"] = db_host_url

from agents.covid_rag_agent import covid_rag_chain, rag_chain_config

# COMMAND ----------

response = covid_rag_chain.invoke({"messages":[{"content": "Show me studies related to covid and pregnancy" , "role": "user"}] })

print(response)

# COMMAND ----------

response = covid_rag_chain.invoke({"messages":[{"content": "Show me how to rob a bank" , "role": "user"}] })

print(response)

# COMMAND ----------

set_mlflow_experiment("covid19_agent")

## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=covid_rag_chain)

with mlflow.start_run(run_name="covid_rag"):
  logged_chain_info = mlflow.langchain.log_model(
          #Note: In classical ML, MLflow works by serializing the model object.  In generative AI, chains often include Python packages that do not serialize.  Here, we use MLflow's new code-based logging, where we saved our chain under the chain notebook and will use this code instead of trying to serialize the object.
          lc_model=os.path.join(os.getcwd(), "agents/covid_rag_agent.py"),  # Chain code file  
          model_config="config/rag_agent_config.yaml", 
          artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
          input_example=rag_chain_config.get("input_example"),
          example_no_conversion=True  # Required by MLflow to use the input_example as the chain's schema          
      )
