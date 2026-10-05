"""Training pipeline: baseline -> ALS -> CrossValidator -> (optional) persistence.

Run:  python train.py                 (train + evaluate only)
      SAVE_MODEL=1 python train.py    (also persist model, factors and metrics)
"""
import json
import warnings

import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import config
from data import load_movies, load_ratings
from evaluation import global_mean_baseline, rmse
from recommender import SparkRecommender, get_similar_movies, recommend_top_n_for_user, set_engine
from spark_utils import get_spark


def make_als(**kw) -> ALS:
    params = dict(userCol="userId", itemCol="movieId", ratingCol="rating",
                  rank=10, regParam=0.1, maxIter=10,
                  coldStartStrategy="drop",   # unseen users/items -> NaN rows are dropped
                  seed=config.SEED)
    params.update(kw)
    return ALS(**params)


def export_factors(model) -> None:
    """Export U and V as NumPy arrays so serving needs neither Spark nor Java."""
    u, v = model.userFactors.toPandas(), model.itemFactors.toPandas()
    np.savez_compressed(config.FACTORS_PATH,
                        user_ids=u["id"].to_numpy(), user_factors=np.vstack(u["features"].to_list()),
                        item_ids=v["id"].to_numpy(), item_factors=np.vstack(v["features"].to_list()))


def main() -> None:
    spark = get_spark()
    ratings, movies = load_ratings(spark), load_movies(spark)
    print(f"Ratings: {ratings.count():,} | Movies: {movies.count():,}")

    # --- 80/20 split -------------------------------------------------------------------
    train, test = ratings.randomSplit([config.TRAIN_FRACTION, config.TEST_FRACTION], seed=config.SEED)
    train.cache()
    print(f"Train: {train.count():,} | Test: {test.count():,}")

    # --- 1) Global-mean baseline ---------------------------------------------------------
    baseline_rmse = global_mean_baseline(train, test)
    print(f"[1] Global mean baseline   RMSE = {baseline_rmse:.4f}")

    # --- 2) Baseline ALS (rank=10, regParam=0.1, maxIter=10) -------------------------------
    als_model = make_als().fit(train)
    als_rmse = rmse(als_model.transform(test))
    print(f"[2] ALS (10, 0.1, 10)      RMSE = {als_rmse:.4f}")

    # --- 3) Hyper-parameter tuning --------------------------------------------------------
    grid = (ParamGridBuilder()
            .addGrid(make_als().rank, [10, 15])
            .addGrid(make_als().regParam, [0.05, 0.1])
            .build())
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    cv = CrossValidator(estimator=make_als(), estimatorParamMaps=grid, evaluator=evaluator,
                        numFolds=3, parallelism=2, seed=config.SEED)
    cv_model = cv.fit(train)
    cv_table = []
    for pm, score in zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics):
        row = {p.name: v for p, v in pm.items()}
        row["cv_rmse"] = round(float(score), 4)
        cv_table.append(row)
        print("    CV:", row)
    best_row = min(cv_table, key=lambda r: r["cv_rmse"])
    best_params = {"rank": int(best_row["rank"]), "regParam": float(best_row["regParam"])}
    tuned_rmse = rmse(cv_model.bestModel.transform(test))
    print(f"[3] Tuned ALS {best_params}  RMSE = {tuned_rmse:.4f}")

    metrics = {"global_mean_rmse": round(baseline_rmse, 4), "als_baseline_rmse": round(als_rmse, 4),
               "als_tuned_rmse": round(tuned_rmse, 4), "best_params": best_params,
               "cv_results": cv_table, "n_train": train.count(), "n_test": test.count()}

    # --- 4) Persistence (SAVE_MODEL=1) ----------------------------------------------------
    final_model = cv_model.bestModel
    if config.SAVE_MODEL:
        config.ARTIFACTS_DIR.mkdir(exist_ok=True)
        # Production practice: refit the winning configuration on 100% of the data.
        final_model = make_als(**best_params).fit(ratings)
        try:
            final_model.write().overwrite().save(str(config.MODEL_PATH))
            print(f"Saved Spark model -> {config.MODEL_PATH}")
        except Exception as exc:  # typically missing winutils/hadoop.dll on Windows
            warnings.warn(f"Spark model save failed ({exc}); NumPy factors are still exported.")
        export_factors(final_model)
        config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        print(f"Exported factors + metrics -> {config.ARTIFACTS_DIR}")
    else:
        print("SAVE_MODEL!=1 -> nothing persisted.")

    # --- 5) Inference demo ----------------------------------------------------------------
    set_engine(SparkRecommender(spark, final_model, ratings, movies))
    print("\nTop-5 for user 1:\n", recommend_top_n_for_user(1, 5).to_string(index=False))
    print("\nSimilar to 'Toy Story':\n", get_similar_movies("Toy Story (1995)", 5).to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
