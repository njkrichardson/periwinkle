import argparse
import os 
import pathlib 
from typing import *

import numpy as np

# --- optional imports 
try: 
    import torch
    tensor: type = torch.Tensor
    layer: type = torch.nn.Module
    network_t: type = layer
except ModuleNotFoundError: 
    pass

# --- aggregate types for numerics 
ndarray: type = np.ndarray 

# --- string types 
path_t: type = Union[os.PathLike, pathlib.Path, str]

# --- namespaces 
namespace: type = argparse.Namespace
