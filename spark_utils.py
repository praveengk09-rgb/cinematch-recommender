"""Spark session factory (driver memory, Windows winutils noise, log level)."""
import os
import sys
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession

import config

# log4j2 config (Spark >= 3.3): root=ERROR and silence the Hadoop winutils/native-lib chatter.
_LOG4J2 = """\
rootLogger.level = error
rootLogger.appenderRef.stdout.ref = console
appender.console.type = Console
appender.console.name = console
appender.console.target = SYSTEM_ERR
appender.console.layout.type = PatternLayout
appender.console.layout.pattern = %d{HH:mm:ss} %p %c{1}: %m%n
logger.shell.name = org.apache.hadoop.util.Shell
logger.shell.level = off
logger.native.name = org.apache.hadoop.util.NativeCodeLoader
logger.native.level = off
"""


def _prepare_windows_env() -> None:
    """Point HADOOP_HOME at a stub dir so Hadoop stops searching for winutils.exe."""
    if not sys.platform.startswith("win"):
        return
    if "HADOOP_HOME" not in os.environ:
        stub = Path(tempfile.gettempdir()) / "hadoop_stub"
        (stub / "bin").mkdir(parents=True, exist_ok=True)
        os.environ["HADOOP_HOME"] = str(stub)
    os.environ.setdefault("hadoop.home.dir", os.environ["HADOOP_HOME"])


def get_spark(app_name: str = "MovieRecALS") -> SparkSession:
    _prepare_windows_env()
    # Make workers use the same interpreter as the driver (avoids Python-not-found on Windows).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    log_cfg = Path(tempfile.gettempdir()) / "recsys_log4j2.properties"
    log_cfg.write_text(_LOG4J2)

    spark = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", config.DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", "16")   # default 200 is overkill for 100K rows
        .config("spark.default.parallelism", "16")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.extraJavaOptions", f"-Dlog4j.configurationFile={log_cfg.as_uri()}")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
