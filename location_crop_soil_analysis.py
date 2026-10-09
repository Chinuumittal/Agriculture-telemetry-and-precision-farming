from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Location-specific agriculture summary:
# one row per location + crop + soil pH band.
# Reads the existing HDFS CSV and writes a small aggregate to a NEW HDFS path.
# No cache/persist and no deletion of existing HDFS data.

INPUT_PATH = "hdfs://master:9000/agriculture/input/agriculture_telemetry.csv"
OUTPUT_PATH = (
    "hdfs://master:9000/user/spark/agriculture_analysis/"
    "location_crop_soil_20261009_v1"
)

spark = (
    SparkSession.builder
    .appName("LocationCropSoilAnalysis")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("Reading agriculture telemetry from HDFS...")
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)
    .option("mode", "PERMISSIVE")
    .csv(INPUT_PATH)
)

required_columns = [
    "location", "crop", "latitude", "longitude", "ph",
    "soil_moisture", "soil_temperature", "nitrogen",
    "phosphorus", "potassium", "temperature", "expected_yield_kg",
]
missing = [name for name in required_columns if name not in df.columns]
if missing:
    raise ValueError("Input CSV is missing required columns: " + ", ".join(missing))

# Cast only fields needed for this aggregation.
numeric_columns = [
    "latitude", "longitude", "ph", "soil_moisture",
    "soil_temperature", "nitrogen", "phosphorus",
    "potassium", "temperature", "expected_yield_kg",
]
for name in numeric_columns:
    df = df.withColumn(name, F.col(name).cast("double"))

# Create understandable pH bands consistent with the earlier soil analysis.
df = df.withColumn(
    "soil_ph_range",
    F.when(F.col("ph").isNull(), F.lit("Unknown"))
     .when(F.col("ph") < 5.5, F.lit("Below 5.5"))
     .when(F.col("ph") < 6.5, F.lit("5.5 to below 6.5"))
     .when(F.col("ph") < 7.5, F.lit("6.5 to below 7.5"))
     .otherwise(F.lit("7.5 and above"))
)

# Filter records with usable location and crop names. Keep null measurements
# in other columns; Spark avg() ignores nulls for each measure.
df = df.filter(
    F.col("location").isNotNull()
    & (F.trim(F.col("location")) != "")
    & F.col("crop").isNotNull()
    & (F.trim(F.col("crop")) != "")
)

print("Aggregating by location, crop, and soil pH range...")
summary = (
    df.groupBy("location", "crop", "soil_ph_range")
    .agg(
        F.count("*").alias("record_count"),
        F.avg("latitude").alias("average_latitude"),
        F.avg("longitude").alias("average_longitude"),
        F.avg("ph").alias("average_ph"),
        F.avg("soil_moisture").alias("average_soil_moisture"),
        F.avg("soil_temperature").alias("average_soil_temperature"),
        F.avg("nitrogen").alias("average_nitrogen"),
        F.avg("phosphorus").alias("average_phosphorus"),
        F.avg("potassium").alias("average_potassium"),
        F.avg("temperature").alias("average_air_temperature"),
        F.avg("expected_yield_kg").alias("average_yield_kg"),
        F.min("expected_yield_kg").alias("minimum_yield_kg"),
        F.max("expected_yield_kg").alias("maximum_yield_kg"),
    )
    .orderBy("location", "crop", "soil_ph_range")
)

# This output path is intentionally new. Spark will fail rather than overwrite
# an existing path, protecting prior results.
print(f"Writing aggregate to: {OUTPUT_PATH}")
(
    summary.coalesce(1)
    .write
    .mode("errorifexists")
    .option("header", True)
    .csv(OUTPUT_PATH)
)

print("Location/crop/soil aggregation completed successfully.")
print(f"Output path: {OUTPUT_PATH}")
spark.stop()
