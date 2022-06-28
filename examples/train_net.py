import argparse 
import time 

import torch 
import torch.nn as nn 

from data_utils import simulate_data, get_default_config
from logging_utils import setup_logger, level_from_args
from nnets import ForecastingNet, ForecastingConfig 
from train_utils import configure_device
from type_aliases import * 

parser: namespace = argparse.ArgumentParser(description="""
        This example script can be launched to train an 
        end-to-end forecasting model. 
        """)

# --- diagnostics 
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--debug", action="store_true")

# --- optimization 
parser.add_argument("--num_epochs", type=int, default=1, help="Number of (full) passes through the dataset.")

# --- platform 
parser.add_argument("--cuda", action="store_true", help="Enable GPU training (if device is available).")

args: namespace = parser.parse_args() 

def main(args: namespace): 
    # --- initialize the platform 
    device: str = configure_device(args.cuda)

    # --- instantiate the network 
    net: network_t = ForecastingNet(ForecastingConfig()).to(device) 
    args.log.info(f"instantiated network of type: {net.model_type}")

    # --- laod the dataset 
    args.log.info("started loading the dataset")
    dataset: DataConfig = simulate_data(save=True)
    args.log.info("finished loading the dataset")

    # --- configure the optimization parameters 
    args.log.info("configuring optimizer")
    criterion = nn.MSELoss()
    learning_rate: float = 1e-2
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # --- train the network 
    net.train() 
    args.log.info("started training")
    start_time: float = time.time()
    for epoch in range(args.num_epochs): 
        epoch_loss: float = 0. 
        num_batches: int = 0 # TODO can I get this from dataset.dataloader?
        
        for i, batch in enumerate(dataset.dataloader): 
            time_series, targets = batch 
            out: tensor = net(time_series)
            batch_loss: float = criterion(out, targets)

            # --- gradient step 
            optimizer.zero_grad() 
            batch_loss.backward() 
            optimizer.step() 

            # --- logging 
            epoch_loss += batch_loss
            num_batches += 1

        args.log.info(f"Epoch [{epoch:03d}/{args.num_epochs:03d}] --- LOSS: {(epoch_loss/num_batches):0.3f}")
    run_time: float = time.time() - start_time
    args.log.info(f"finished training in: {run_time:0.2f}s")

if __name__=="__main__": 
    vars(args)["log"] = setup_logger(__name__, level=level_from_args(args)) 
    main(args)
