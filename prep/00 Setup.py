# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Data and Other Resources

# COMMAND ----------

# MAGIC %run ./utils/init

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Catalog, Schema and Volume

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{data_folder}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Copy COVID Dataset to a Volume

# COMMAND ----------

from databricks.sdk import WorkspaceClient

#using workspace api instead of dbutils to copy files
#this is necessary for serverless support
w = WorkspaceClient()
with w.workspace.download(f"/{project_root_path}/resources/{covid_data_file_name}") as f:
  data = f.read()

w.files.upload(file_path=f"/Volumes/{catalog}/{schema}/{data_folder}/{covid_data_file_name}", contents=data)


# COMMAND ----------


