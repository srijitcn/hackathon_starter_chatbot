# Databricks notebook source
# MAGIC %pip install -q databricks-vectorsearch==0.40 databricks-sdk==0.38.0 databricks-agents==0.6.0 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk.service.dashboards import GenieAPI
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
genie_api = w.genie

# COMMAND ----------


