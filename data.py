"""Data ingestion with explicit schemas."""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (FloatType, IntegerType, LongType, StringType,
                               StructField, StructType)

import config

RATINGS_SCHEMA = StructType([
    StructField("userId", IntegerType(), False),
    StructField("movieId", IntegerType(), False),
    StructField("rating", FloatType(), False),
    StructField("timestamp", LongType(), True),
])

MOVIES_SCHEMA = StructType([
    StructField("movieId", IntegerType(), False),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True),
])


def load_ratings(spark: SparkSession) -> DataFrame:
    """Read ratings.csv, drop malformed/null rows and enforce the target types."""
    if not config.RATINGS_CSV.exists():
        raise FileNotFoundError(f"{config.RATINGS_CSV} not found. Run: python download_data.py")
    df = (spark.read.option("header", True).option("mode", "DROPMALFORMED")
          .schema(RATINGS_SCHEMA).csv(str(config.RATINGS_CSV)))
    return (df.dropna(subset=["userId", "movieId", "rating"])
              .select(F.col("userId").cast("int"), F.col("movieId").cast("int"),
                      F.col("rating").cast("float"), "timestamp"))


def load_movies(spark: SparkSession) -> DataFrame:
    """Read movies.csv (titles may contain quoted commas, hence multiLine/escape)."""
    if not config.MOVIES_CSV.exists():
        raise FileNotFoundError(f"{config.MOVIES_CSV} not found. Run: python download_data.py")
    return (spark.read.option("header", True).option("multiLine", True)
            .option("quote", '"').option("escape", '"')
            .schema(MOVIES_SCHEMA).csv(str(config.MOVIES_CSV)))
