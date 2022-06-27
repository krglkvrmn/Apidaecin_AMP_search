import os
import shutil

LOGS_BASE_DIR = 'runs'


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
    shutil.rmtree(LOGS_BASE_DIR, ignore_errors=True)
    os.makedirs(LOGS_BASE_DIR, exist_ok=True)


def tb_run(run_name: str) -> str:
    """
    Create tensorboard run instance (bind it with log_dir)
    :param run_name: Name of model run
    :type run_name: str
    :return: Path to run logs
    :rtype: str
    """
    return os.path.join(LOGS_BASE_DIR, run_name)
