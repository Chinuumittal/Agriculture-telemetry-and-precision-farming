from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, trim, when

spark = (
    SparkSession.builder
    .appName("AgricultureSoilAnalysis")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

input_path = (
    "hdfs://master:9000/agriculture/input/"
    "agriculture_telemetry.csv"
)

output_path = (
    "hdfs://master:9000/user/spark/agriculture_analysis/"
    "soil_summary_20261009"
)

# Read only the fields needed for this analysis.
df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv(input_path)
    .select(
        "crop",
        "ph",
        "soil_moisture",
        "soil_temperature",
        "nitrogen",
        "phosphorus",
        "potassium",
        "expected_yield_kg"
    )
)

# Convert numeric columns explicitly.
numeric_columns = [
    "ph",
    "soil_moisture",
    "soil_temperature",
    "nitrogen",
    "phosphorus",
    "potassium",
    "expected_yield_kg"
]

for name in numeric_columns:
    df = df.withColumn(name, col(name).cast("double"))

df = df.withColumn("crop", trim(col("crop")))

# Analyze soil pH bands, including nutrient and yield summaries.
valid = df.filter(
    col("crop").isNotNull()
    & (col("crop") != "")
    & col("ph").isNotNull()
)

banded = valid.withColumn(
    "soil_ph_range",
    when(col("ph") < 5.5, "Below 5.5")
    .when(col("ph") < 6.5, "5.5 to below 6.5")
    .when(col("ph") < 7.5, "6.5 to below 7.5")
    .otherwise("7.5 and above")
)

summary = (
    banded.groupBy("crop", "soil_ph_range")
    .agg(
        count("*").alias("record_count"),
        avg("ph").alias("average_ph"),
        avg("soil_moisture").alias("average_soil_moisture"),
        avg("soil_temperature").alias("average_soil_temperature"),
        avg("nitrogen").alias("average_nitrogen"),
        avg("phosphorus").alias("average_phosphorus"),
        avg("potassium").alias("average_potassium"),
        avg("expected_yield_kg").alias("average_yield_kg")
    )
)

# Fail safely if this output folder already exists.
(
    summary.coalesce(1)
    .write
    .mode("errorifexists")
    .option("header", "true")
    .csv(output_path)
)

print("SOIL ANALYSIS COMPLETED")
print("Output:", output_path)

spark.stop()
