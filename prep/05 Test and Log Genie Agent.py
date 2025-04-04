# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

import os

os.environ["GENIE_AGENT_CONFIG_FILE"] = "config/genie_agent_config.yaml"

# COMMAND ----------

from agents.genie_agent import genie_agent, genie_config
from langchain_core.messages import HumanMessage

response = genie_agent.invoke({"messages":[HumanMessage(content="tell me how many COVID trials are in Recruiting status")] })

print(response)

# COMMAND ----------


