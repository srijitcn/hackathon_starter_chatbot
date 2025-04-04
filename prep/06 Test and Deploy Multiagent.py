# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

import os
import mlflow 

multi_agent_config = mlflow.models.ModelConfig(development_config="config/multi_agent_config.yaml")
os.environ[multi_agent_config.get("web_search_api_key_environment_var")] = dbutils.secrets.get("multi_agent","web_search_api_key")
os.environ[multi_agent_config.get("helper_agent_config_environment_var")] = "config/helper_agent_config.yaml"
os.environ[multi_agent_config.get("genie_agent_config_environment_var")] = "config/genie_agent_config.yaml"
os.environ[multi_agent_config.get("rag_agent_config_environment_var")] = "config/rag_agent_config.yaml"
os.environ[multi_agent_config.get("multi_agent_config_environment_var")] = "config/multi_agent_config.yaml"
os.environ["WORKSPACE_URL"] = db_host_url

from agents.multiagent import graph_with_parser, rag_config, helper_config, genie_config
from langchain_core.messages import HumanMessage

# COMMAND ----------

graph_with_parser.invoke({
    "messages": [
        HumanMessage(content="What are covid studies related to pregnancy?")
    ],
    "num_attempts": 0,
    "max_attempts": 5
})

# COMMAND ----------

graph_with_parser.invoke({
    "messages": [
        HumanMessage(content="How many COVID trials are in Recruiting status?")
    ],
    "num_attempts": 0,
    "max_attempts": 5
})

# COMMAND ----------

#get the resources required by the chain
multi_agent_llm_config = multi_agent_config.get("multi_agent_llm_config")
multiagent_llm_endpoint=multi_agent_llm_config.get("llm_endpoint_name")
genie_space_id =  genie_config.get("genie_space_id")
rag_model_endpoint = rag_config.get("llm_endpoint_name")
rag_vector_index_name = rag_config.get("retriever_config").get("vector_search_index")
math_tool_model_endpoint =  helper_config.get("math_tool").get("llm_endpoint_name")

print("########### Databricks Resources:")
print(f"multiagent_llm_endpoint:{multiagent_llm_endpoint}")
print(f"genie_space_id:{genie_space_id}")
print(f"rag_model_endpoint:{rag_model_endpoint}")
print(f"rag_vector_index_name:{rag_vector_index_name}")
print(f"math_tool_model_endpoint:{math_tool_model_endpoint}")


# COMMAND ----------

import os
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    DatabricksGenieSpace
)

model_name = "multi_agent_langgraph"
uc_model_name = f"{catalog}.{schema}.{model_name}"

set_mlflow_experiment("covid19_agent")

with mlflow.start_run(run_name="multi_agent"):
  logged_chain_info = mlflow.langchain.log_model(
          #Note: In classical ML, MLflow works by serializing the model object.  In generative AI, chains often include Python packages that do not serialize.  Here, we use MLflow's new code-based logging, where we saved our chain under the chain notebook and will use this code instead of trying to serialize the object.
          lc_model=os.path.join(os.getcwd(), "agents/multiagent.py"),  # Chain code file  
          model_config="config/multi_agent_config.yaml", 
          artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
          extra_pip_requirements=["mlflow",
                                  "databricks-langchain",
                                  "langchain-community",
                                  "langgraph",
                                  "beautifulsoup4"],
          code_paths = ["agents", "config"],
          input_example=multi_agent_config.get("input_example"),
          example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema     
          # Specify resources for automatic authentication passthrough
          resources = [
                DatabricksServingEndpoint(endpoint_name=multiagent_llm_endpoint),
                DatabricksServingEndpoint(endpoint_name=math_tool_model_endpoint),
                DatabricksServingEndpoint(endpoint_name=rag_model_endpoint),
                DatabricksGenieSpace(genie_space_id=genie_space_id),
                DatabricksVectorSearchIndex(index_name=rag_vector_index_name)
          ]          
  )

# COMMAND ----------

#register the model
mlflow.set_registry_uri("databricks-uc")
# Register to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=uc_model_name)

# COMMAND ----------

from databricks import agents

env_vars = {
    multi_agent_config.get("web_search_api_key_environment_var") : dbutils.secrets.get("multi_agent","web_search_api_key"),
    multi_agent_config.get("helper_agent_config_environment_var") : "config/helper_agent_config.yaml",
    multi_agent_config.get("genie_agent_config_environment_var") : "config/genie_agent_config.yaml",
    multi_agent_config.get("rag_agent_config_environment_var") : "config/rag_agent_config.yaml",
    multi_agent_config.get("multi_agent_config_environment_var") : "config/multi_agent_config.yaml"
}

deployment_info = agents.deploy(
    model_name=uc_model_name,
    model_version=uc_registered_model_info.version,
    scale_to_zero=True,
    environment_vars=env_vars
)
