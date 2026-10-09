
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("HDFSConnectionTest") \
    .getOrCreate()

df = spark.read.text(
    "hdfs://master:9000/wordcount/input/input.txt"
)

df.show(truncate=False)

print("Total lines:", df.count())

spark.stop()
