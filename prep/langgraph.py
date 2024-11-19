# Databricks notebook source
# MAGIC %pip install -U langgraph langchain langchain_experimental databricks-sdk mlflow databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.langchain.autolog()

# COMMAND ----------

from databricks_langchain.genie import GenieAgent

# add your genie space id here
genie_space_id =  "01efa5d3862c1317969a569e3cdfcb1c"
genie_agent = GenieAgent(genie_space_id, "Genie", description="This Genie space has data on COVID Clinical Trials")
     

# COMMAND ----------

#test
genie_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "How many trials are in Recruiting status?",
            }
        ]
    })

# COMMAND ----------


