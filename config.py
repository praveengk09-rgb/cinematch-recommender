"""Central configuration. No heavy imports here so serving stays lightweight."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "ml-latest-small"))
RATINGS_CSV = DATA_DIR / "ratings.csv"
MOVIES_CSV = DATA_DIR / "movies.csv"

ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "als_model"        # native Spark ALSModel
FACTORS_PATH = ARTIFACTS_DIR / "factors.npz"    # exported U and V for Spark-free serving
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"   # shown in the Streamlit UI

SEED = 42
TRAIN_FRACTION, TEST_FRACTION = 0.8, 0.2
RATING_MIN, RATING_MAX = 0.5, 5.0
DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")

# Persistence is controlled by an environment variable: SAVE_MODEL=1
SAVE_MODEL = os.getenv("SAVE_MODEL", "0") == "1"
