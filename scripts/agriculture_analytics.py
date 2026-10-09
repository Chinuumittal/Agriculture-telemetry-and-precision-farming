from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, min as spark_min, max as spark_max, count,
    when, trim
)

spark = SparkSession.builder.appName(
    "AgricultureTelemetryAnalytics"
).getOrCreate()

spark.sparkContext.setLogLevel("WARN")

input_path = (
    "hdfs://master:9000/agriculture/input/"
    "agriculture_telemetry.csv"
)
output_base = (
    "hdfs://master:9000/user/spark/agriculture_analysis/"
    "step2_20261009"
)

# Read only the columns required for our analysis.
# inferSchema=False avoids expensive schema inference.
raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("mode", "PERMISSIVE")
    .csv(input_path)
    .select(
        "location", "crop", "latitude", "longitude",
        "temperature", "humidity", "rainfall",
        "soil_moisture", "soil_temperature", "ph",
        "nitrogen", "phosphorus", "potassium",
        "expected_yield_kg", "irrigation_status",
        "irrigation_method", "water_usage_liters"
    )
)

numeric_columns = [
    "latitude", "longitude", "temperature", "humidity",
    "rainfall", "soil_moisture", "soil_temperature", "ph",
    "nitrogen", "phosphorus", "potassium",
    "expected_yield_kg", "water_usage_liters"
]

df = raw
for name in numeric_columns:
    df = df.withColumn(name, trim(col(name)).cast("double"))

df = (
    df.withColumn("crop", trim(col("crop")))
      .withColumn("location", trim(col("location")))
      .withColumn("irrigation_status", trim(col("irrigation_status")))
      .withColumn("irrigation_method", trim(col("irrigation_method")))
      .filter(col("crop").isNotNull() & (col("crop") != ""))
      
)

# Materialize the narrow, selected dataset once.
total_records = df.count()
print(f"TOTAL VALID CROP RECORDS: {total_records}")

def save_csv(data, folder):
    (
        data.write
        .mode("overwrite")
        .option("header", "true")
        .csv(f"{output_base}/{folder}")
    )
    print(f"Saved: {output_base}/{folder}")

# 1. Crop-wise yield statistics
crop_summary = (
    df.filter(col("expected_yield_kg").isNotNull())
      .groupBy("crop")
      .agg(
          count("*").alias("record_count"),
          avg("expected_yield_kg").alias("average_yield_kg"),
          spark_min("expected_yield_kg").alias("minimum_yield_kg"),
          spark_max("expected_yield_kg").alias("maximum_yield_kg")
      )
      .orderBy(col("average_yield_kg").desc())
)
save_csv(crop_summary, "crop_yield")

# 2. Yield by temperature range
temperature_data = (
    df.filter(
        col("temperature").isNotNull()
        & col("expected_yield_kg").isNotNull()
    )
    .withColumn(
        "temperature_range",
        when(col("temperature") < 15, "Below 15 C")
        .when(col("temperature") < 25, "15-24.9 C")
        .when(col("temperature") < 35, "25-34.9 C")
        .otherwise("35 C and above")
    )
)
temperature_summary = (
    temperature_data.groupBy("temperature_range")
    .agg(
        count("*").alias("record_count"),
        avg("temperature").alias("average_temperature"),
        avg("expected_yield_kg").alias("average_yield_kg")
    )
    .orderBy("temperature_range")
)
save_csv(temperature_summary, "temperature_analysis")

# 3. Yield by soil pH range
soil_data = (
    df.filter(
        col("ph").isNotNull()
        & col("expected_yield_kg").isNotNull()
    )
    .withColumn(
        "soil_ph_range",
        when(col("ph") < 5.5, "Acidic: below 5.5")
        .when(col("ph") < 6.5, "5.5-6.49")
        .when(col("ph") < 7.5, "6.5-7.49")
        .otherwise("7.5 and above")
    )
)
soil_summary = (
    soil_data.groupBy("soil_ph_range")
    .agg(
        count("*").alias("record_count"),
        avg("ph").alias("average_ph"),
        avg("soil_moisture").alias("average_soil_moisture"),
        avg("nitrogen").alias("average_nitrogen"),
        avg("phosphorus").alias("average_phosphorus"),
        avg("potassium").alias("average_potassium"),
        avg("expected_yield_kg").alias("average_yield_kg")
    )
    .orderBy("soil_ph_range")
)
save_csv(soil_summary, "soil_analysis")

# 4. Highest-yielding locations
location_summary = (
    df.filter(
        col("location").isNotNull()
        & (col("location") != "")
        & col("expected_yield_kg").isNotNull()
    )
    .groupBy("location")
    .agg(
        count("*").alias("record_count"),
        avg("expected_yield_kg").alias("average_yield_kg"),
        avg("temperature").alias("average_temperature"),
        avg("rainfall").alias("average_rainfall"),
        avg("soil_moisture").alias("average_soil_moisture"),
        avg("water_usage_liters").alias("average_water_usage_liters"),
        avg("latitude").alias("latitude"),
        avg("longitude").alias("longitude")
    )
    .orderBy(col("average_yield_kg").desc())
)
save_csv(location_summary, "location_analysis")

# 5. Irrigation analysis
irrigation_summary = (
    df.filter(
        col("irrigation_status").isNotNull()
        & col("expected_yield_kg").isNotNull()
    )
    .groupBy("irrigation_status", "irrigation_method")
    .agg(
        count("*").alias("record_count"),
        avg("expected_yield_kg").alias("average_yield_kg"),
        avg("water_usage_liters").alias("average_water_usage_liters")
    )
    .orderBy(col("average_yield_kg").desc())
)
save_csv(irrigation_summary, "irrigation_analysis")

df.unpersist()
spark.stop()
print("ALL AGRICULTURE ANALYSES COMPLETED SUCCESSFULLY")

