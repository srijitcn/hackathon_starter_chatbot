# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

from databricks_langchain.genie import GenieAgent

# add your genie space id here
genie_space_id =  "01efae624eb11421a468187406487ff4"
genie_agent = GenieAgent(genie_space_id, "Genie", description="This Genie Agent will have all data about COVID Trials and related articles")

# COMMAND ----------

response = genie_agent.invoke({"messages":[{"content": "tell me how many COVID trials are in Recruiting status" , "role": "user"}] })

print(response)
