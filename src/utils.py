from datetime import datetime
from functools import lru_cache
import os
import pickle

from type_aliases import * 

@lru_cache(maxsize=1)
def get_source_dir() -> path_t:
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def get_project_dir() -> path_t:
    return os.path.dirname(get_source_dir())


def get_project_subdirectory(name: str) -> path_t:
    abs_path: path_t = os.path.join(get_project_dir(), name)
    if os.path.exists(abs_path) is False:
        os.mkdir(abs_path)
    return abs_path


def save_object(obj: Any, location: path_t):
    if location[-4:] != ".pkl":
        location += ".pkl"
    with open(location, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(location: path_t) -> Any:
    with open(location, "rb") as handle:
        result = pickle.load(handle)
    return result

@lru_cache(maxsize=1)
def get_figures_dir() -> path_t:
    return get_project_subdirectory("figures")


@lru_cache(maxsize=1)
def get_data_directory() -> path_t:
    return get_project_subdirectory("data")


@lru_cache(maxsize=1)
def get_log_dir() -> path_t:
    return get_project_subdirectory("logs")

def get_now_str() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def human_bytes_str(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB")
    power = 2**10

    for unit in units:
        if num_bytes < power:
            return f"{num_bytes:.1f} {unit}"

        num_bytes /= power

    return f"{num_bytes:.1f} TB"
