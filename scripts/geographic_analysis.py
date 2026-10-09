from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, trim
)

spark = (
    SparkSession.builder
    .appName("AgricultureGeographicAnalysis")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

input_path = (
    "hdfs://master:9000/agriculture/input/"
    "agriculture_telemetry.csv"
)

output_path = (
    "hdfs://master:9000/user/spark/agriculture_analysis/"
    "geographic_summary_20261009"
)

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv(input_path)
    .select(
        "location",
        "crop",
        "latitude",
        "longitude",
        "expected_yield_kg"
    )
    .withColumn("location", trim(col("location")))
    .withColumn("crop", trim(col("crop")))
    .withColumn("latitude", col("latitude").cast("double"))
    .withColumn("longitude", col("longitude").cast("double"))
    .withColumn(
        "expected_yield_kg",
        col("expected_yield_kg").cast("double")
    )
)

# Keep records with valid coordinates and yield.
valid = df.filter(
    col("location").isNotNull()
    & (col("location") != "")
    & col("latitude").between(-90, 90)
    & col("longitude").between(-180, 180)
    & col("expected_yield_kg").isNotNull()
)

# Summarize each location and crop.
summary = (
    valid.groupBy("location", "crop")
    .agg(
        count("*").alias("record_count"),
        avg("expected_yield_kg").alias("average_yield_kg"),
        avg("latitude").alias("latitude"),
        avg("longitude").alias("longitude")
    )
    .orderBy("location", "crop")
)

(
    summary.coalesce(1)
    .write
    .mode("errorifexists")
    .option("header", "true")
    .csv(output_path)
)

print("GEOGRAPHIC ANALYSIS COMPLETED")
print("Output:", output_path)

spark.stop()
