# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

import os

os.environ["GENIE_AGENT_CONFIG_FILE"] = "config/genie_agent_config.yaml"

# COMMAND ----------

from agents.genie_agent import genie_agent, genie_config

response = genie_agent.invoke({"messages":[{"content": "tell me how many COVID trials are in Recruiting status" , "role": "user"}] })

print(response)

# COMMAND ----------


