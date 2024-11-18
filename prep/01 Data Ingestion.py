# Databricks notebook source
# MAGIC %md
# MAGIC #Ingesting COVID Data into Unity Catalog

# COMMAND ----------

# MAGIC %run ./utils/init

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ingesting raw csv file into delta table

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType, LongType
from datetime import datetime, date

#function to convert given dates tring to date type
#we see dates with and without day of the month
def convert_date_str_to_date(date_str:str):
  if date_str is not None:
    date_str = date_str.replace(",","")
    try:
      return datetime.strptime(date_str, "%B %d %Y").date()
    except:
      try:
        return datetime.strptime(date_str, "%B %Y").date()
      except:
        return None
  else:
    return None

# COMMAND ----------

#test
assert convert_date_str_to_date("January 5, 2020")==date(2020, 1, 5)
assert convert_date_str_to_date("January, 2020")==date(2020, 1, 1)
assert convert_date_str_to_date("January 2020")==date(2020, 1, 1)
assert convert_date_str_to_date("2020")==None

# COMMAND ----------

covid_file_path=f"/Volumes/{catalog}/{schema}/{data_folder}/{covid_data_file_name}"

covid_df_schema = StructType([
    StructField("Rank",IntegerType(), nullable=False),
    StructField("NCT Number",StringType(), nullable=False),   
    StructField("Title",StringType(), nullable=False),
    StructField("Acronym",StringType(), nullable=False),
    StructField("Status",StringType(), nullable=False),
    StructField("Study Results",StringType(), nullable=False),
    StructField("Conditions",StringType(), nullable=False),
    StructField("Interventions",StringType(), nullable=True),
    StructField("Outcome Measures",StringType(), nullable=False),
    StructField("Sponsor/Collaborators",StringType(), nullable=False),
    StructField("Gender",StringType(), nullable=False),
    StructField("Age",StringType(), nullable=False),
    StructField("Phases",StringType(), nullable=True),
    StructField("Enrollment",LongType(), nullable=False),
    StructField("Funded Bys",StringType(), nullable=False),
    StructField("Study Type",StringType(), nullable=False),
    StructField("Study Designs",StringType(), nullable=False),
    StructField("Other IDs",StringType(), nullable=False),
    StructField("Start Date",StringType(), nullable=True),
    StructField("Primary Completion Date",StringType(), nullable=True),
    StructField("Completion Date",StringType(), nullable=True),
    StructField("First Posted",StringType(), nullable=True),
    StructField("Results First Posted",StringType(), nullable=True),
    StructField("Last Update Posted",StringType(), nullable=True),
    StructField("Locations",StringType(), nullable=True),
    StructField("Study Documents",StringType(), nullable=True),
    StructField("URL",StringType(), nullable=False)   
])

date_columns = [
    "Start Date",
    "Primary Completion Date",
    "Completion Date",
    "First Posted",
    "Results First Posted",
    "Last Update Posted",
]

convert_date_str_to_date_udf = udf(convert_date_str_to_date, DateType())

covid_df = (spark
          .read
          .option("header", "true")
          .option("delimiter", ",")
          .schema(covid_df_schema)
          .csv(covid_file_path)
)

for column in date_columns:
    covid_df = covid_df.withColumn(column, convert_date_str_to_date_udf(column))


# COMMAND ----------

#cleaning up column names
for column in covid_df.columns:
    covid_df = covid_df.withColumnRenamed(column, column.replace(" ", "_").replace("/", "_").lower() )

# COMMAND ----------

display(covid_df)
#Read more about the columns here: https://www.kaggle.com/datasets/parulpandey/covid19-clinical-trials-dataset/data

# COMMAND ----------

covid_df.count()

# COMMAND ----------

covid_df.printSchema()

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{covid_data_table_name}")

spark.catalog.createTable(f"{catalog}.{schema}.{covid_data_table_name}", schema=covid_df.schema)

covid_df.write.mode("append").saveAsTable(f"{catalog}.{schema}.{covid_data_table_name}")


# COMMAND ----------

spark.sql(f"ALTER TABLE {catalog}.{schema}.{covid_data_table_name} ALTER COLUMN nct_number SET NOT NULL")

spark.sql(f"ALTER TABLE {catalog}.{schema}.{covid_data_table_name} ADD CONSTRAINT {covid_data_table_name}_pk PRIMARY KEY( nct_number )")

spark.sql(f"ALTER TABLE {catalog}.{schema}.{covid_data_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true) ")

# COMMAND ----------


