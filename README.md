# Scalable Movie Recommendation System: PySpark ALS

Pipeline: MovieLens CSV -> explicit schema -> 80/20 split -> global-mean baseline -> ALS -> CrossValidator
(rank [10,15] x regParam [0.05,0.1]) -> optional persistence -> FastAPI + Streamlit.

```
movie_recsys/
├── recsys/
│   ├── config.py        paths, seed, clip range, SAVE_MODEL flag
│   ├── spark_utils.py   SparkSession (4g driver, ERROR logs, winutils suppression)
│   ├── data.py          explicit schemas + loaders
│   ├── evaluation.py    clipping, RMSE, global-mean baseline
│   ├── core.py          ItemSpace: exact cosine similarity, title resolution (no Spark)
│   ├── recommender.py   SparkRecommender: recommend_top_n_for_user / get_similar_movies
│   └── serving.py       NumPy recommender from exported factors (no Spark)
├── train.py             full training + tuning + persistence
├── api.py               FastAPI: /recommend, /similar
├── app.py               Streamlit UI
└── download_data.py
```

## 1. Setup
Requires Python 3.10+ and **Java 11 or 17** (`java -version`).
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
```

## 2. Train
```bash
python train.py                    # train and evaluate only
SAVE_MODEL=1 python train.py       # also saves artifacts/ (Linux/macOS)
```
Windows PowerShell: `$env:SAVE_MODEL=1; python train.py`  | cmd: `set SAVE_MODEL=1 && python train.py`

With `SAVE_MODEL=1` it writes `artifacts/als_model` (native Spark), `factors.npz` (U and V for Spark-free serving)
and `metrics.json`. The winning config is refit on 100% of the data before saving.

## 3. REST API
```bash
uvicorn api:app --port 8000
curl "http://localhost:8000/recommend?user_id=1&top_n=10"
curl "http://localhost:8000/similar?movie_id=1&top_k=5"
```
Swagger docs: http://localhost:8000/docs. Unknown users/movies return HTTP 404.

## 4. Streamlit UI
```bash
streamlit run app.py
```

## 5. Deploy on Streamlit Community Cloud
Spark/Java is not needed at serving time. Train locally, then commit `app.py`, `recsys/`,
`artifacts/factors.npz`, `artifacts/metrics.json`, `data/ml-latest-small/{ratings,movies}.csv`,
and make the repo's root `requirements.txt` the contents of `requirements-serving.txt`.
Choose `app.py` as the entry point.

## Windows notes
* The winutils warning is silenced (stub `HADOOP_HOME` + log4j2 filter). Spark runs fine without it.
* Saving the native Spark model on Windows may need `winutils.exe`/`hadoop.dll`; if it fails you get a
  warning and `factors.npz` is still exported, so the API and UI keep working.
