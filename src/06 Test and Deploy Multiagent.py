# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

import os
import mlflow 

multi_agent_config_root = mlflow.models.ModelConfig(development_config="config/multi_agent_config.yaml")
multi_agent_config = multi_agent_config_root.get("multi_agent_config")
websearch_api_env_var = multi_agent_config_root.get("helper_agent_config").get("web_search_tool").get("api_key_environment_var")

os.environ[websearch_api_env_var] = dbutils.secrets.get("multi_agent","web_search_api_key")
os.environ["DATABRICKS_HOST"] = db_host_url
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get("multi_agent","pat")
os.environ["HELPER_AGENT_CONFIG_FILE"] = "config/helper_agent_config.yaml"
os.environ["GENIE_AGENT_CONFIG_FILE"] = "config/genie_agent_config.yaml"
os.environ["RAG_AGENT_CONFIG_FILE"] = "config/rag_agent_config.yaml"
os.environ["MULTI_AGENT_CONFIG_FILE"] = "config/multi_agent_config.yaml"

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
multi_agent_llm_config = multi_agent_config.get("llm_config")
multiagent_llm_endpoint=multi_agent_llm_config.get("llm_endpoint_name")
genie_space_id =  genie_config.get("genie_space_id")
genie_space_sql_warehouse_id =  genie_config.get("genie_space_sql_warehouse_id")
rag_model_endpoint = rag_config.get("llm_endpoint_name")
rag_vector_index_name = rag_config.get("retriever_config").get("vector_search_index")
math_tool_model_endpoint =  helper_config.get("math_tool").get("llm_endpoint_name")

print("########### Databricks Resources:")
print(f"multiagent_llm_endpoint:{multiagent_llm_endpoint}")
print(f"genie_space_id:{genie_space_id}")
print(f"genie_space_sql_warehouse_id:{genie_space_sql_warehouse_id}")
print(f"rag_model_endpoint:{rag_model_endpoint}")
print(f"rag_vector_index_name:{rag_vector_index_name}")
print(f"math_tool_model_endpoint:{math_tool_model_endpoint}")


# COMMAND ----------

import os
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksVectorSearchIndex,
    DatabricksGenieSpace
)
from mlflow.models.auth_policy import (
  SystemAuthPolicy,
  UserAuthPolicy,
  AuthPolicy
)

model_name = "multi_agent_langgraph"
uc_model_name = f"{catalog}.{schema}.{model_name}"

set_mlflow_experiment("covid19_agent")

system_resources = [
                DatabricksServingEndpoint(endpoint_name=multiagent_llm_endpoint),
                DatabricksServingEndpoint(endpoint_name=math_tool_model_endpoint),
                DatabricksServingEndpoint(endpoint_name=rag_model_endpoint),
                DatabricksGenieSpace(genie_space_id=genie_space_id),
                DatabricksVectorSearchIndex(index_name=rag_vector_index_name),
                DatabricksSQLWarehouse(warehouse_id=genie_space_sql_warehouse_id)
          ]      

# system_auth_policy = SystemAuthPolicy(resources=system_resources) 

# user_auth_policy = UserAuthPolicy(
#     api_scopes=[
#         "serving.serving-endpoints",
#         "vectorsearch.vector-search-endpoints",
#         "vectorsearch.vector-search-indexes",
#         "dashboards.genie"
#     ]
# ) 

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
          resources = system_resources
          # auth_policy=AuthPolicy(
          #   system_auth_policy=system_auth_policy, user_auth_policy=user_auth_policy
          # )  
  )

# COMMAND ----------

#register the model
mlflow.set_registry_uri("databricks-uc")
# Register to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=uc_model_name)

# COMMAND ----------

from databricks import agents

env_vars = {
    websearch_api_env_var : dbutils.secrets.get("multi_agent","web_search_api_key"),
    "DATABRICKS_HOST" : db_host_url,
    "DATABRICKS_TOKEN" : dbutils.secrets.get("multi_agent","pat")
}

deployment_info = agents.deploy(
    model_name=uc_model_name,
    model_version=uc_registered_model_info.version,
    scale_to_zero=True,
    environment_vars=env_vars
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set the configs for a UI application

# COMMAND ----------

endpoint_name = deployment_info.query_endpoint

# Our frontend application will hit the model endpoint we deployed.
# Because dbdemos let you change your catalog and database, let's make sure we deploy the app with the proper endpoint name
yaml_app_config = f"""
command: [
  "streamlit", 
  "run",
  "app.py"
]

env:
  - name: STREAMLIT_BROWSER_GATHER_USAGE_STATS
    value: "false"
  - name: "SERVING_ENDPOINT"
    value: "{deployment_info.endpoint_name}"
"""

try:
    with open('app/app.yaml', 'w') as f:
        f.write(yaml_app_config)
except Exception as e:
    print(e)

