import dataclasses 

import torch 

from logging_utils import setup_logger
from typing import * 

def configure_device(use_gpu: bool=False) -> str: 
    log = setup_logger(__name__) 
    
    if torch.cuda.is_available(): 
        return "cuda" 
    else: 
        log.info(f"GPU not found.")
        return "cpu" 

@dataclasses.dataclass
class TrainConfig: 
    batch_size: Optional[int] = 128 
