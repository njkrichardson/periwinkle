![release_tag](https://img.shields.io/github/v/release/njkrichardson/periwinkle)

## Deep Learning For Financial Forecasting

Periwinkle is a research package for applying deep learning methods to financial forecasting tasks. The package contains utilities for managing and simulating data, 
models for learning and inference implemented in [PyTorch](https://pytorch.org/), and fire-and-forget examples to get started. 

## Contents 
  1. [Installing Dependencies](https://github.com/njkrichardson/periwinkle#installing-dependencies)
  2. [Getting Started](https://github.com/njkrichardson/periwinkle#getting-started) 
  3. [Forecasting Model](https://github.com/njkrichardson/periwinkle#forecasting-model) 
  4. [References](https://github.com/njkrichardson/periwinkle#references) 

---
## Installing Dependencies 

### Configuring Your Path 

The modules under `src` must be visible to your Python3 interpreter to be imported. You can do this by updating your shell's `PYTHONPATH` environment
variable to include this directory. To do this, place the following line into your shell configuration file (e.g., `.bashrc` for bash users or `.zshrc` for 
zsh users) or in a `.envrc` under the top-level project directory for [direnv](https://direnv.net/) users. 

```bash
export PYTHONPATH=$PYTHONPATH:/Path/to/periwinkle/src 
```


### Automated (conda | pip) 
To install the dependencies using either [conda](https://docs.conda.io/en/latest/) or the Python package installer [pip](https://pypi.org/project/pip/), 
execute one of the following in your shell once you've navigated to the top-level project directory: 

```bash
$ conda env create --name=periwinkle --file environment.yml
```

```bash
$ python3 -m pip --requirement=requirements.txt
```

### Manual

Periwinkle requires an installation of Python3.7 or higher, as well as [PyTorch](https://pytorch.org/) ([installation instructions](https://pytorch.org/get-started/locally/)) and [NumPy](https://numpy.org/doc/stable/reference/index.html#reference) ([installation instructions](https://numpy.org/devdocs/user/building.html)). 
Periwinkle was tested against PyTorch 1.11.0 and NumPy 1.23.0. 

## Getting Started 

An example of training the Forecasting Model (using simulated data) is provided in the [examples](https://github.com/njkrichardson/periwinkle/tree/main/examples) directory (`examples/train_net.py`). 
We walk through the example here to get a sense for how it works. 

First, we configure the device we're training on. Pass the `--cuda` flag to utilize GPU training, by default the script uses CPU only. 

```python
# --- initialize the platform 
device: str = configure_device(args.cuda)
```

Next, we instantiate the Forecasting Model (a custom model inheriting from `torch.nn.Module`) and load the dataset. If it's your first time running the script, 
the dataset will be simulated and cached to disk; subsequent invocations will simply load the dataset from this cached serialization. 

```python
# --- instantiate the network 
net: network_t = ForecastingNet(ForecastingConfig()).to(device) 

# --- load the dataset 
dataset: DataConfig = simulate_data(save=True)
```
To use automatic differentiation to compute derivatives of our cost (here, mean-squared error) with respect to the parameters, we need to instantiate an [optimizer object](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)
and with an iterable containing any parameters. 

```python
# --- configure the optimization parameters 
criterion = nn.MSELoss()
learning_rate: float = 1e-2
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
```

Finally, we execute gradient-based optimization of our cost with respect to the parameters. 

```python
# --- train the network 
net.train() 
for epoch in range(args.num_epochs): 
    ...
    for i, batch in enumerate(dataset.dataloader): 
        time_series, targets = batch 
        out: tensor = net(time_series)
        batch_loss: float = criterion(out, targets)

        # --- gradient step 
        optimizer.zero_grad() 
        batch_loss.backward() 
        optimizer.step() 
        ...
```

The default command-line arguments are sufficient to execute the script, but run `python3 examples/train_net.py --help` for usage. 

```bash
$ python3 -m examples/train_net.py --help
usage: train_net.py [-h] [--verbose] [--debug] [--num_epochs NUM_EPOCHS] [--cuda]

This example script can be launched to train an end-to-end forecasting model.

optional arguments:
  -h, --help            show this help message and exit
  --verbose
  --debug
  --num_epochs NUM_EPOCHS
                        Number of (full) passes through the dataset.
  --cuda                Enable GPU training (if device is available).
```

## Forecasting Model 

ForecastingNet is a vanilla discriminative model over multivariate time series. In the application domain, each element of the time series is a real-valued matrix.
One axis indexes financial assets and the other axis indexes per-asset features. The targets are represented as real-valued vectors with length equal to the 
number of financial assets in the input time series. If the input contains 8 assets, the associated target is a real 8-vector representing the returns associated
with the 8 input assets at the subsequent timestep. 

The network consists of two basic submodules: a Transformer encoder ([EmbeddingNet](https://github.com/njkrichardson/periwinkle/blob/497d41853534fb29f7f5b16e73c0f245a9a7280d/src/nnets.py#L73)) and a 
vanilla fully-connected feedforward network ([FeedforwardNet](https://github.com/njkrichardson/periwinkle/blob/497d41853534fb29f7f5b16e73c0f245a9a7280d/src/nnets.py#L35)). 
The feedforward net is implemented independently from the embedding network to anticipate applications in which auxiliary but non-temporal features are provided 
with the input time series. In that case the transformer can be invoked to produce an embedding of the temporally ordered features, and then aggregated (e.g., concatenated) 
with the non-ordered features and processed with a downstream network to produce a final value. 


## References 

### Academic 
  [1] [Attention is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al.
  
### Gentler 
  [2] [Transformers Explained Visually](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)
  
  [3] [What are Transformer Neural Networks?](https://www.youtube.com/watch?v=XSSTuhyAmnI&t=741s)
  
 
