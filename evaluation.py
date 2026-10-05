"""Evaluation helpers: clipping, RMSE, global-mean baseline."""
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import config


def clip_predictions(df: DataFrame, col: str = "prediction") -> DataFrame:
    """Bound predictions strictly inside [0.5, 5.0]."""
    return df.withColumn(col, F.least(F.greatest(F.col(col), F.lit(config.RATING_MIN)),
                                      F.lit(config.RATING_MAX)))


def rmse(predictions: DataFrame) -> float:
    ev = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    return float(ev.evaluate(clip_predictions(predictions)))


def global_mean_baseline(train: DataFrame, test: DataFrame) -> float:
    """Predict mu (the mean training rating) for every test row."""
    mu = train.agg(F.avg("rating")).first()[0]
    return rmse(test.withColumn("prediction", F.lit(float(mu))))
