# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

import os
os.environ["BRAVE_API_KEY"] = dbutils.secrets.get("multi_agent","web_search_api_key")
os.environ["HELPER_AGENT_CONFIG_FILE"] = "config/helper_agent_config.yaml"

# COMMAND ----------


from agents.helper_agent import helper_chain

# COMMAND ----------

response = helper_chain.invoke({"messages":[{"content": "whats the capital of france" , "role": "user"}] })

print(response)

# COMMAND ----------

response = helper_chain.invoke({"messages":[{"content": "How many florida can fit in India?" , "role": "user"}] })

# COMMAND ----------

#get the resources required by the chain
agent_config = mlflow.models.ModelConfig(development_config="config/helper_agent_config.yaml")

math_tool_model_endpoint =  agent_config.get("math_tool").get("llm_endpoint_name")

print("########### Databricks Resources:")
print(f"math_tool_model_endpoint:{math_tool_model_endpoint}")


# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint

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
          example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema          
          # Specify resources for automatic authentication passthrough
          resources = [
                DatabricksServingEndpoint(endpoint_name=math_tool_model_endpoint),
          ]  
      )

# COMMAND ----------


