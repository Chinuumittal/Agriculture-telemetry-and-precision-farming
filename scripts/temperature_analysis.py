from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, trim, when

spark = (
    SparkSession.builder
    .appName("AgricultureTemperatureAnalysis")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

input_path = (
    "hdfs://master:9000/agriculture/input/"
    "agriculture_telemetry.csv"
)

output_path = (
    "hdfs://master:9000/user/spark/agriculture_analysis/"
    "temperature_summary_20261009"
)

# Read only the columns needed for this analysis.
df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv(input_path)
    .select("temperature", "expected_yield_kg")
    .withColumn("temperature", col("temperature").cast("double"))
    .withColumn(
        "expected_yield_kg",
        col("expected_yield_kg").cast("double")
    )
    .filter(
        col("temperature").isNotNull()
        & col("expected_yield_kg").isNotNull()
    )
)

# Create temperature bands.
banded = df.withColumn(
    "temperature_range",
    when(col("temperature") < 15, "Below 15 C")
    .when(col("temperature") < 25, "15 to below 25 C")
    .when(col("temperature") < 35, "25 to below 35 C")
    .otherwise("35 C and above")
)

summary = (
    banded.groupBy("temperature_range")
    .agg(
        count("*").alias("record_count"),
        avg("temperature").alias("average_temperature"),
        avg("expected_yield_kg").alias("average_yield_kg")
    )
)

(
    summary.coalesce(1)
    .write
    .mode("errorifexists")
    .option("header", "true")
    .csv(output_path)
)

print("TEMPERATURE ANALYSIS COMPLETED")
print("Output:", output_path)

spark.stop()
