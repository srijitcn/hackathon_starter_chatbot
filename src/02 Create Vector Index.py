# Databricks notebook source
# MAGIC %md
# MAGIC #Create Vector Index

# COMMAND ----------

# MAGIC %md
# MAGIC There are a few free text columns that we can use for question answering along with other structured information
# MAGIC - title
# MAGIC - interventions
# MAGIC - outcome_measures
# MAGIC - study_designs
# MAGIC
# MAGIC To keep it simple, We will create a vector index on `title` field for now

# COMMAND ----------

# MAGIC %run ./utils/init

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create a Vector Search endpoint
# MAGIC vector Search Endpoint serves the vector search index. You can query and update the endpoint using the REST API or the SDK. Endpoints scale automatically to support the size of the index or the number of concurrent requests. See [Create a vector search endpoint](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint) for instructions.

# COMMAND ----------

import databricks
from databricks.vector_search.client import VectorSearchClient

#name for the vector search endpoint
vector_search_endpoint_name = "covid_trials_vs_endpoint" 

#We are using an embedding endpoint available in Databricks Workspace
#If needed we can use custom embedding endpoints as well
embedding_endpoint_name = "databricks-bge-large-en" 

#Define the source tables, index name and key fields
covid_trial_title_index_source_data_table = f"{catalog}.{schema}.{covid_data_table_name}"
covid_trial_title_index_source_data_table_id_field = "nct_number"  
covid_trial_title_index_source_data_table_text_field = "title" 
covid_trial_title_index_vector_index_name = f"{covid_trial_title_index_source_data_table}_title_index"


# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from datetime import timedelta
import time
#create the vector search endpoint if it does not exist
#same endpoint can be used to serve both the indexes
vsc = VectorSearchClient(disable_notice=True)

try:
    vsc.create_endpoint(name=vector_search_endpoint_name,
                        endpoint_type="STANDARD")
    
    time.sleep(5)

    vsc.wait_for_endpoint(name=vector_search_endpoint_name,
                                timeout=timedelta(minutes=60),
                                verbose=True)
    
    print(f"Endpoint named {vector_search_endpoint_name} is ready.")

    ep = vsc.get_endpoint(name=vector_search_endpoint_name)

except Exception as e:
    if "already exists" in str(e):
        print(f"Endpoint named {vector_search_endpoint_name} already exists.")
        ep = vsc.get_endpoint(name=vector_search_endpoint_name)
    else:
        raise e

# COMMAND ----------

#giving all workspace users USE access to the endpoint

from databricks.sdk.service import iam
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
w.permissions.set(request_object_type="vector-search-endpoints",
                  request_object_id=ep["id"],
                  access_control_list=[
                        iam.AccessControlRequest(group_name="users",
                                                   permission_level=iam.PermissionLevel.CAN_MANAGE)
                      ])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test Embedding Endpoint
# MAGIC
# MAGIC We will use the existing `databricks-bge-large-en` endpoint for embeddings

# COMMAND ----------

import mlflow
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------

[ep for ep in client.list_endpoints() if ep["name"]==embedding_endpoint_name]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test the embeddings endpoint

# COMMAND ----------

client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create Vector Search Index
# MAGIC The vector search index is created from a Delta table and is optimized to provide real-time approximate nearest neighbor searches. The goal of the search is to identify documents that are similar to the query. Vector search indexes appear in and are governed by Unity Catalog. To learn more about creating Vector Indexes, visit this [link](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html). 
# MAGIC
# MAGIC We will now create the vector indexes. Our vector index will be of `Delta Sync Index` type. [[Read More](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index)] 
# MAGIC
# MAGIC We will use a Sync Mode of `TRIGGERED` as our table updates are not happening frequently and sync latency is not an issue for us. [[Read More](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index:~:text=embedding%20table.-,Sync%20mode%3A,-Continuous%20keeps%20the)]

# COMMAND ----------

# MAGIC %md
# MAGIC #####Create covid_trials_title Vector Index

# COMMAND ----------

# MAGIC %md
# MAGIC **NOTE:** Below section creates a vector search index and does an initial sync. Some time this could take longer and the cell execution might timeout. You can re-run the cell to finish to completion

# COMMAND ----------


try:
  covid_trial_title_index = vsc.create_delta_sync_index_and_wait(
    endpoint_name=vector_search_endpoint_name,
    index_name=covid_trial_title_index_vector_index_name,
    source_table_name=covid_trial_title_index_source_data_table,
    primary_key=covid_trial_title_index_source_data_table_id_field,
    embedding_source_column=covid_trial_title_index_source_data_table_text_field,
    embedding_model_endpoint_name=embedding_endpoint_name,
    pipeline_type="TRIGGERED",
    verbose=True
  )
except Exception as e:
    if "already exists" in str(e):
        print(f"Index named {vector_search_endpoint_name} already exists.")
        covid_trial_title_index = vsc.get_index(vector_search_endpoint_name, 
                                                covid_trial_title_index_vector_index_name)
    else:
        raise e

# COMMAND ----------

spark.sql(f"GRANT SELECT ON TABLE {covid_trial_title_index_vector_index_name} TO `account users` ")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Quick Test of Index

# COMMAND ----------

results = covid_trial_title_index.similarity_search(
  query_text="What are pregnancy related complications in COVID-19 patients",
  columns=["nct_number", "title","interventions","url"],
  num_results=1
)

if results["result"]["row_count"] >0:
  display(results["result"]["data_array"])
else:
  print("No records")
