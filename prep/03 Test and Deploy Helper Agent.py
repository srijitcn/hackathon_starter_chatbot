# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

import os
os.environ["brave_api_key"] = "BSAi72gapKd22Y6aX3_krqpv9Cx3QAr"

from agents.helper_agent import helper_chain

# COMMAND ----------



response = helper_chain.invoke({"messages":[{"content": "How many Florida can fit in area of India" , "role": "user"}] })

print(response)

# COMMAND ----------


