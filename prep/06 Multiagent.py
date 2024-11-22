# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

import os

os.environ["BRAVE_API_KEY"] = dbutils.secrets.get("multi_agent","web_search_api_key")
os.environ["VECTOR_SEARCH_CLIENT_ID"] = dbutils.secrets.get("multi_agent","vector_search_client_id")
os.environ["VECTOR_SEARCH_CLIENT_SECRET"] = dbutils.secrets.get("multi_agent","vector_search_client_secret")
os.environ["WORKSPACE_URL"] = db_host_url


from agents.multiagent import graph_with_parser, multi_agent_config

# COMMAND ----------

graph_with_parser.invoke({"messages":[{"content": "How many florida can fit iin India" , "role": "user"}] })

# COMMAND ----------

graph_with_parser.invoke({"messages":[{"content": "What are covid studies realted to pregnancy?" , "role": "user"}] })

# COMMAND ----------

import os

set_mlflow_experiment("covid19_agent")

with mlflow.start_run(run_name="multi_agent"):
  logged_chain_info = mlflow.langchain.log_model(
          #Note: In classical ML, MLflow works by serializing the model object.  In generative AI, chains often include Python packages that do not serialize.  Here, we use MLflow's new code-based logging, where we saved our chain under the chain notebook and will use this code instead of trying to serialize the object.
          lc_model=os.path.join(os.getcwd(), "agents/multiagent.py"),  # Chain code file  
          model_config="config/multi_agent_config.yaml", 
          artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
          extra_pip_requirements=["mlflow",
                                  "langchain-community",
                                  "langgraph",
                                  "beautifulsoup4"],
          code_paths = ["agents" ],
          input_example=multi_agent_config.get("input_example"),
          example_no_conversion=True  # Required by MLflow to use the input_example as the chain's schema     
  )

# COMMAND ----------

#loaded_chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
#loaded_chain.invoke(multi_agent_config.get("input_example"))

# COMMAND ----------

#register the model
model_name = "multi_agent"
uc_model_name = f"{catalog}.{schema}.{model_name}"
mlflow.set_registry_uri("databricks-uc")
# Register to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=uc_model_name)

# COMMAND ----------

from databricks import agents

api_key_env_var = multi_agent_config.get("web_search_tool").get("api_key_environment_var").upper()
client_id_environment_var = multi_agent_config.get("retriever_config").get("client_id_environment_var").upper()
client_secret_environment_var = multi_agent_config.get("retriever_config").get("client_secret_environment_var").upper()
workspace_url_environment_var = multi_agent_config.get("retriever_config").get("workspace_url_environment_var").upper()

deployment_info = agents.deploy(
    model_name=uc_model_name,
    model_version=uc_registered_model_info.version,
    scale_to_zero=True,
    environment_vars={
        api_key_env_var : dbutils.secrets.get("multi_agent","web_search_api_key"),
        client_id_environment_var : dbutils.secrets.get("multi_agent","vector_search_client_id"),
        client_secret_environment_var : dbutils.secrets.get("multi_agent","vector_search_client_secret"),
        workspace_url_environment_var : db_host_url
    }
)


# COMMAND ----------

print(api_key_env_var)

# COMMAND ----------


