# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

from agents.covid_rag_agent import covid_rag_chain, chain_config

# COMMAND ----------

response = covid_rag_chain.invoke({"messages":[{"content": "Show me studies related to covid and pregnancy" , "role": "user"}] })

print(response)

# COMMAND ----------

response = covid_rag_chain.invoke({"messages":[{"content": "Show me how to rob a bank" , "role": "user"}] })

print(response)

# COMMAND ----------

import os

set_mlflow_experiment("covid19_agent")

with mlflow.start_run(run_name="covid_rag"):
  logged_chain_info = mlflow.langchain.log_model(
          #Note: In classical ML, MLflow works by serializing the model object.  In generative AI, chains often include Python packages that do not serialize.  Here, we use MLflow's new code-based logging, where we saved our chain under the chain notebook and will use this code instead of trying to serialize the object.
          lc_model=os.path.join(os.getcwd(), "agents/covid_rag_agent.py"),  # Chain code file  
          model_config="config/rag_agent_config.yaml", 
          artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
          input_example=chain_config.get("input_example"),
          example_no_conversion=True  # Required by MLflow to use the input_example as the chain's schema          
      )


# COMMAND ----------

# Test the chain locally
loaded_chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
loaded_chain.invoke(chain_config.get("input_example"))

# COMMAND ----------

#register the model
model_name = "covid_rag_agent"
uc_model_name = f"{catalog}.{schema}.{model_name}"
# Register to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=uc_model_name)

# COMMAND ----------

from databricks import agents

deployment_info = agents.deploy(model_name=uc_model_name,
                                model_version=uc_registered_model_info.version,
                                scale_to_zero=True)

# COMMAND ----------


