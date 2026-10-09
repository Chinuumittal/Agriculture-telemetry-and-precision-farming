from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, trim

spark = (
    SparkSession.builder
    .appName("AgricultureIrrigationAnalysis")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

input_path = (
    "hdfs://master:9000/agriculture/input/"
    "agriculture_telemetry.csv"
)

output_path = (
    "hdfs://master:9000/user/spark/agriculture_analysis/"
    "irrigation_summary_20261009"
)

# Read only the three columns needed for this analysis.
df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv(input_path)
    .select(
        "irrigation_status",
        "irrigation_method",
        "expected_yield_kg",
        "water_usage_liters"
    )
)

# Explicitly convert numeric columns.
df = (
    df.withColumn(
        "expected_yield_kg",
        col("expected_yield_kg").cast("double")
    )
    .withColumn(
        "water_usage_liters",
        col("water_usage_liters").cast("double")
    )
    .withColumn(
        "irrigation_status",
        trim(col("irrigation_status"))
    )
    .withColumn(
        "irrigation_method",
        trim(col("irrigation_method"))
    )
)

# Group records by irrigation status and method.
summary = (
    df.filter(
        col("irrigation_status").isNotNull()
        & (col("irrigation_status") != "")
        & col("irrigation_method").isNotNull()
        & (col("irrigation_method") != "")
    )
    .groupBy("irrigation_status", "irrigation_method")
    .agg(
        count("*").alias("record_count"),
        avg("expected_yield_kg").alias("average_yield_kg"),
        avg("water_usage_liters").alias(
            "average_water_usage_liters"
        )
    )
    .orderBy("irrigation_status", "irrigation_method")
)

# The grouped result should be small; do not cache the input.
(
    summary.coalesce(1)
    .write
    .mode("errorifexists")
    .option("header", "true")
    .csv(output_path)
)

print("IRRIGATION ANALYSIS COMPLETED")
print("Output:", output_path)

spark.stop()
