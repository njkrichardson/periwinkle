import dataclasses
import os

import torch 
from torch.utils.data import Dataset, DataLoader

from logging_utils import setup_logger
from type_aliases import * 
from utils import get_data_directory, load_object, save_object

@dataclasses.dataclass
class DataConfig: 
    """Configuration dataclass for datasets. Encapsulates 
    the underlying torch.utils.data.Dataset and the 
    torch.utils.data.DataLoader, along with several 
    configuration options.
    """
    data_path: path_t
    dataset: Dataset
    dataloader: DataLoader

class CustomDataset(Dataset): 
    """Custom dataset class encapsulating the financial 
    predictors (time series and auxiliary predictors) and 
    targets.
    """
    def __init__(self, data_path: path_t=None): 
        self.data_path = os.path.join(get_data_directory(), "data.pkl") if data_path is None else data_path
        self.data: dict = load_object(self.data_path)

    def __len__(self) -> int: 
        return self.data["inputs"].shape[0]

    def __getitem__(self, idx: int) -> Tuple[tensor, tensor, tensor]: 
        time_series: tensor = self.data["inputs"][idx]
        target: tensor = self.data["targets"][idx] 

        if type(time_series) == ndarray: 
            return torch.from_numpy(time_series), torch.from_numpy(target)
        elif type(time_series) == tensor: 
            return time_series,  target
        else: 
            raise NotImplementedError

def create_dataloader(dataset: Dataset, shuffle: bool=True, batch_size: int=64, num_workers: int=0) -> DataLoader: 
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_default_config() -> DataConfig: 
    data_path: path_t = os.path.join(get_data_directory(), "data.pkl")
    dataset: CustomDataset = CustomDataset()
    dataloader: DataLoader = create_dataloader(dataset) 
    config: DataConfig = DataConfig(
            data_path=data_path, 
            dataset=dataset, 
            dataloader=dataloader,
            )
    return config 

def simulate_data(num_examples: int=8_192, sequence_length: int=16, num_assets: int=8, asset_dim: int=32, save: bool=False) -> dict: 
    log = setup_logger(__name__) 
    save_path: path_t = os.path.join(get_data_directory(), "data.pkl")

    if os.path.exists(save_path): 
        log.info(f"found dataset at: {save_path}")
        return get_default_config() 

    log.info(f"did not find existing dataset at: {save_path}")
    log.info("generating...")

    time_series: tensor = torch.randn(num_examples, sequence_length, num_assets, asset_dim)
    targets: tensor = torch.randn(num_examples, num_assets)
    dataset: dict = dict(
            inputs=time_series, 
            targets=targets
            )

    if save: 
        save_object(dataset, save_path) 
        log.info(f"saved dataset to: {save_path}")

    return get_default_config()
