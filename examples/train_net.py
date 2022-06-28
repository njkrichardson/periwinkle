import argparse 

import torch 
import torch.nn as nn 

from data_utils import simulate_data, get_default_config
from logging_utils import setup_logger, level_from_args
from nnets import ForecastingNet, ForecastingConfig 
from type_aliases import * 

parser: namespace = argparse.ArgumentParser(description="""
        This example script can be launched to train an 
        end-to-end forecasting model. 
        """)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--debug", action="store_true")
args: namespace = parser.parse_args() 

def main(args: namespace): 
    # --- instantiate the network 
    import pdb; pdb.set_trace()
    net: network_t = ForecastingNet(ForecastingConfig())

    # --- laod the dataset 
    dataset: DataConfig = simulate_data(save=True)

    # --- configure the optimization parameters 
    criterion = nn.MSELoss()
    learning_rate: float = 1e-2
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # --- train the network 
    net.train() 
    for i, batch in enumerate(dataset.dataloader): 
        time_series_batch, auxiliary_batch, targets_batch = batch 
        out: tensor = net((time_series_batch, auxiliary_batch))
        loss: float = criterion(out, targets)

        # --- gradient step 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        # --- logging 
        args.log.info(f"Loss: {loss}")

if __name__=="__main__": 
    vars(args)["log"] = setup_logger(__name__, level=level_from_args(args)) 
    main(args)
