# Databricks notebook source
from sklearn.datasets import load_iris
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

iris = load_iris(as_frame=True)

pdf = iris.frame.copy()
pdf.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

pdf.insert(0, "iris_id", range(1, len(pdf) + 1))

df = spark.createDataFrame(pdf)

# COMMAND ----------

features_df = df.drop("species")
labels_df = df.select("iris_id", "species")

feature_table = f"{catalog}.{schema}.iris_features"
label_table = f"{catalog}.{schema}.iris_labels"

spark.sql(f"DROP TABLE IF EXISTS {feature_table}")

fe = FeatureEngineeringClient()

fe.create_table(
    name=feature_table,
    primary_keys=["iris_id"],
    df=features_df,
    description="Iris flower measurements"
)

labels_df.write.mode("overwrite").saveAsTable(label_table)
