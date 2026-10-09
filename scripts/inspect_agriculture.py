from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("AgricultureDatasetSample") \
    .getOrCreate()

path = "hdfs://master:9000/agriculture/input/agriculture_telemetry.csv"

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .option("mode", "PERMISSIVE") \
    .csv(path)

print("========== SCHEMA ==========")
df.printSchema()

print("========== FIRST 10 ROWS ==========")
df.show(10, truncate=False)

print("========== COLUMN COUNT ==========")
print(len(df.columns))

print("========== PARTITIONS ==========")
print(df.rdd.getNumPartitions())

spark.stop()
