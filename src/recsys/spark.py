import os

from pyspark.sql import SparkSession

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")


def get_spark(app_name: str, driver_memory: str = "8g") -> SparkSession:
    """Start (or reuse) a local Spark session that uses all CPU cores."""
    spark = (
        SparkSession.builder
        .master("local[*]")                          # run on this Mac, using all cores
        .appName(app_name)                           # a name shown in Spark's own web page
        .config("spark.driver.memory", driver_memory)  # memory Spark may use
        .config("spark.driver.host", "127.0.0.1")      # talk to itself on this machine only,
        .config("spark.driver.bindAddress", "127.0.0.1")  # not over your Wi-Fi network
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")          # hide Spark's very chatty log messages
    return spark