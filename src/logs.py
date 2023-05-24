import os
import shutil
import sys
from pathlib import Path

from loguru import logger

from src.config import read_yaml_config, get_value_from_config

config = read_yaml_config("config.yaml")


LOGS_DIR = Path(get_value_from_config(config, keys=["logs", "logs_dir"], default="logs"))
LOGS_LEVEL = get_value_from_config(config, keys=["logs", "logs_level"], default=15)
LOGS_MODE = get_value_from_config(config, keys=["logs", "logs_mode"], default="jupyter")

stream_sink = sys.stderr if LOGS_MODE == "pipeline" else sys.stdout

logger.remove(0)
logger.level("TRAINING", no=15, color="<fg #3eeeff>")
logger.level("PREDICTION", no=17, color="<fg #b0f200>")
logger.add(stream_sink, level=LOGS_LEVEL, colorize=True)
logger.add(LOGS_DIR / "{time:HH:mm:ss}" / "log.log", level=LOGS_LEVEL, rotation="1 GB", colorize=True)
logger.add(LOGS_DIR / "{time:HH:mm:ss}" / "error.log", level="ERROR", rotation="1 GB", colorize=True)


TB_LOGS_BASE_DIR = 'runs'


def reinit_tensorboard_local(logs_dir: str, clear_log: bool = True):
    """
    Reinitialize tensorboard locally
    :param logs_dir: directory where particular run is stored
    :type logs_dir: str
    :param clear_log: whether to clear previous results or not
    :type clear_log: bool
    """
    if clear_log:
        shutil.rmtree(logs_dir, ignore_errors=True)
        os.makedirs(logs_dir, exist_ok=True)


def delete_all_runs():
    """
    Delete all logs used by tensorboard
    """
    shutil.rmtree(TB_LOGS_BASE_DIR, ignore_errors=True)
    os.makedirs(TB_LOGS_BASE_DIR, exist_ok=True)


def tb_run(run_name: str) -> str:
    """
    Create tensorboard run instance (bind it with log_dir)
    :param run_name: Name of model run
    :type run_name: str
    :return: Path to run logs
    :rtype: str
    """
    return os.path.join(TB_LOGS_BASE_DIR, run_name)
