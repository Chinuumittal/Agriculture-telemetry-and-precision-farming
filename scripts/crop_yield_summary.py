from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, min, max, count, trim

spark = SparkSession.builder.appName("CropYieldSummary").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

input_path = "hdfs://master:9000/agriculture/input/agriculture_telemetry.csv"
output_path = "hdfs://master:9000/user/spark/agriculture_analysis/crop_yield_small_20261009"

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv(input_path)
    .select("crop", "expected_yield_kg")
    .withColumn("crop", trim(col("crop")))
    .withColumn("expected_yield_kg", col("expected_yield_kg").cast("double"))
    .filter(
        col("crop").isNotNull()
        & (col("crop") != "")
        & col("expected_yield_kg").isNotNull()
    )
)

result = (
    df.groupBy("crop")
    .agg(
        count("*").alias("record_count"),
        avg("expected_yield_kg").alias("average_yield_kg"),
        min("expected_yield_kg").alias("minimum_yield_kg"),
        max("expected_yield_kg").alias("maximum_yield_kg")
    )
    .orderBy("crop")
)

result.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

print("Crop-wise yield summary saved to:", output_path)
spark.stop()
