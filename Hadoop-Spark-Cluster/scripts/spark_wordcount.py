
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, col

spark = SparkSession.builder \
    .appName("SparkWordCount") \
    .getOrCreate()

input_path = "hdfs://master:9000/wordcount/input/input.txt"
output_path = "hdfs://master:9000/wordcount/spark-output"

lines = spark.read.text(input_path)

words = lines.select(
    explode(split(col("value"), r"\s+")).alias("word")
)

frequencies = (
    words
    .filter(col("word") != "")
    .groupBy(lower(col("word")).alias("word"))
    .count()
    .orderBy("word")
)

frequencies.show(truncate=False)

frequencies.write.mode("overwrite").csv(output_path, header=True)

print("Spark WordCount completed.")
print("Output saved to:", output_path)

spark.stop()
