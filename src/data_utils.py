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
        self.data_path = os.path.join(get_data_directory(), "data.pkl") if archive_location is None else archive_location
        self.data: dict = load_object(self.data_path)

    def __len__(self) -> int: 
        return self.data["inputs"]["time_series"].shape[0]

    def __getitem__(self, idx: int) -> Tuple[tensor, tensor, tensor]: 
        _time_series, _auxiliary = self.data["inputs"]["time_series"][idx], self.data["inputs"]["auxiliary"][idx]
        _target: ndarray = self.data["targets"][idx] 
        return torch.from_numpy(_time_series), torch.from_numpy(_auxiliary), torch.from_numpy(_target)

def create_dataloader(dataset: Dataset, archive_location: path_t=None, shuffle: bool=True, batch_size: int=64, num_workers: int=0) -> DataLoader: 
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_default_config() -> DataConfig: 
    dataset: CustomDataset = CustomDataset()
    dataloader: DataLoader = create_dataloader(dataset) 
    config: DataConfig = DataConfig(
            data_path=data_path, 
            dataset=dataset, 
            dataloader=dataloader,
            )
    return config 

def simulate_data(num_timesteps: int=10_000, time_series_dim: int=64, auxiliary_dim: int=16, save: bool=False) -> dict: 
    import pdb; pdb.set_trace()
    log = setup_logger(__name__) 
    save_path: path_t = os.path.join(get_data_directory(), "data.pkl")

    if os.path.exists(save_path): 
        log.info(f"found dataset at: {save_path}")
        return get_default_config() 

    log.info(f"did not find existing dataset at: {save_path}")
    log.info("generating...")

    time_series: tensor = torch.randn(num_timesteps, time_series_dim)
    auxiliary: tensor = torch.randn(num_timesteps, auxiliary_dim)
    targets: tensor = torch.randn(num_timesteps, time_series_dim)
    dataset: dict = dict(
            time_series=time_series, 
            auxiliary=auxiliary, 
            targets=targets
            )

    if save: 
        save_object(dataset, save_path) 
        log.info(f"saved dataset to: {save_path}")
    else: 
        return get_default_config()
