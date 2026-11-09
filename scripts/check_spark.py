from recsys.config import load_config
from recsys.spark import get_spark


def main():
    cfg = load_config()
    spark = get_spark("check-spark")

    print("Spark version:", spark.version)
    print("Cores Spark can use:", spark.sparkContext.defaultParallelism)

    train = spark.read.parquet(str(cfg["paths"]["processed"] / "train.parquet"))
    train.printSchema()
    print(f"Training ratings: {train.count():,}")

    spark.stop()


if __name__ == "__main__":
    main()