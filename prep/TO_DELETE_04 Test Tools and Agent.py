# Databricks notebook source
# MAGIC %run "./03 Create Tools and Agent"

# COMMAND ----------

#test
retriever_config = RetrieverConfig(vector_search_endpoint_name="covid_trials_vs_endpoint",
                            vector_index_name=f"{catalog}.{schema}.{covid_data_table_name}_title_index",
                            vector_index_id_column="nct_number",
                            num_documents_to_retrieve=3,
                            retrieve_columns=["title","interventions","url"])

covid_trial_rag = CovidTrialTitleRAG(
  model_endpoint_name="databricks-meta-llama-3-1-70b-instruct",
  retriever_config=retriever_config).get()

response = covid_trial_rag.run({
  "filters":{"status":"Recruiting"},
  #"filters":{},
  "query":"What are the studies related to pregnancy and covid"
})

print(response)

