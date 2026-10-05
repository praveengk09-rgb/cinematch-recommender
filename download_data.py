"""Download and unpack MovieLens 'latest-small' (~100K ratings) into ./data."""
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEST = Path(__file__).resolve().parent / "data"

if __name__ == "__main__":
    DEST.mkdir(exist_ok=True)
    print(f"Downloading {URL} ...")
    
    # Create unverified SSL context to resolve local certificate verification failure on Windows
    ssl_context = ssl._create_unverified_context()
    
    with urllib.request.urlopen(URL, context=ssl_context) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DEST)
    print(f"Done -> {DEST / 'ml-latest-small'}")
