
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, min, max, sum as spark_sum, when, trim
)

spark = SparkSession.builder \
    .appName("AgricultureDataValidation") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

path = "hdfs://master:9000/agriculture/input/agriculture_telemetry.csv"

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .option("mode", "PERMISSIVE") \
    .csv(path)

# Cast only fields required for numerical analysis
df = df.withColumn(
    "expected_yield_kg", col("expected_yield_kg").cast("double")
).withColumn(
    "latitude", col("latitude").cast("double")
).withColumn(
    "longitude", col("longitude").cast("double")
)

print("\n========== DATASET SIZE ==========")
print("Total columns:", len(df.columns))

# Run all missing-value checks in one aggregation
print("\n========== MISSING VALUE CHECK ==========")
checks = df.agg(
    count("*").alias("total_records"),
    spark_sum(
        when(col("crop").isNull() | (trim(col("crop")) == ""), 1).otherwise(0)
    ).alias("missing_crop"),
    spark_sum(
        when(col("expected_yield_kg").isNull(), 1).otherwise(0)
    ).alias("missing_or_invalid_yield"),
    spark_sum(
        when(col("latitude").isNull(), 1).otherwise(0)
    ).alias("missing_or_invalid_latitude"),
    spark_sum(
        when(col("longitude").isNull(), 1).otherwise(0)
    ).alias("missing_or_invalid_longitude")
)

checks.show(truncate=False)

# Crop-wise yield analysis
print("\n========== CROP-WISE YIELD SUMMARY ==========")
crop_summary = df.groupBy("crop").agg(
    count("*").alias("record_count"),
    avg("expected_yield_kg").alias("average_yield_kg"),
    min("expected_yield_kg").alias("minimum_yield_kg"),
    max("expected_yield_kg").alias("maximum_yield_kg")
).orderBy("crop")

crop_summary.show(50, truncate=False)

# Save the result to a new HDFS location

output_path = "hdfs://master:9000/user/spark/agriculture_analysis/crop_summary_20261009"


print("\n========== SAVING CROP SUMMARY ==========")
crop_summary.write \
    .mode("errorifexists") \
    .option("header", "true") \
    .csv(output_path)

print("Summary saved to:", output_path)

spark.stop()
