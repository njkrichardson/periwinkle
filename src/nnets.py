import dataclasses
import math 

import torch 
from torch import nn 

from type_aliases import * 

@dataclasses.dataclass
class ForecastingConfig: 
    """Configuration dataclass for the end-to-end
    forecasting network.
    """
    sequence_length: Optional[int] = 16
    asset_dimension: Optional[int] = 32
    num_assets: Optional[int] = 8

    # --- time series embedding 
    embedding_input_dimension: Optional[int] = 128

    # --- transformer 
    model_dimension: Optional[int] = 64 
    num_heads: Optional[int] = 4
    hidden_dimension: Optional[int] = 16 
    num_layers: Optional[int] = 4 

    # --- aggregation net
    mlp_input_dimension: Optional[int] = 1024
    output_dimension: Optional[int] = 8 
    nonlinearity: Optional[str] = "relu" 

    # --- auxiliary parameters 
    dropout_rate: Optional[float] = 0.

class FeedForwardNet(nn.Module): 
    """Vanilla fully-connected multi-layer 
    perceptron (i.e., a feedforward neural net). 

    TODO: make layer sizes parametric 
    """
    def __init__(self, config: ForecastingConfig): 
        super(FeedForwardNet, self).__init__()
        nonlinearity: layer = nn.ReLU() if config.nonlinearity == "relu" else NotImplementedError
        self.net: layer = nn.Sequential(
                nn.Linear(config.mlp_input_dimension, 128), nonlinearity, 
                nn.Linear(128, config.output_dimension)
                )
        self.config = config

    def forward(self, x: tensor) -> tensor: 
        out: tensor = self.net(x.view(-1, self.config.sequence_length * self.config.model_dimension)) 
        return out 

class PositionalEncoding(nn.Module): 
    """Fourier positional encoding as suggested 
    in the original paper.
    """
    def __init__(self, model_dimension: int, dropout_rate: float, max_length: int=4096): 
        super(PositionalEncoding, self).__init__()
        self.dropout: layer = nn.Dropout(dropout_rate)

        position: tensor = torch.arange(max_length).unsqueeze(1) 
        normalization: tensor = torch.exp(torch.arange(0, model_dimension, 2) * (-math.log(10_000.) / model_dimension))
        encoding: tensor = torch.zeros(max_length, 1, model_dimension)
        encoding[:, 0, 0::2] = torch.sin(position * normalization) 
        encoding[:, 0, 1::2] = torch.cos(position * normalization) 
        self.register_buffer("encoding", encoding)

    def forward(self, x: tensor) -> tensor: 
        x = x + self.encoding[:x.size(0)]
        return self.dropout(x) 

class EmbeddingNet(nn.Module): 
    """Time-series embedding network for use 
    with sequence data. The architecture is a 
    transformer encoder. 
    """
    def __init__(self, config: ForecastingConfig): 
        super(EmbeddingNet, self).__init__() 
        self.encoder: layer = nn.Linear(config.embedding_input_dimension, config.model_dimension)
        self.positional_encoder: layer = PositionalEncoding(config.model_dimension, config.dropout_rate)
        self.transformer_encoder: layer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(config.model_dimension, config.num_heads, config.hidden_dimension, config.dropout_rate), 
                config.num_layers
                )
        self.config = config 

        self.initialize_weights() 

    def initialize_weights(self) -> None: 
        initialization_range: float = 1e-1
        self.encoder.weight.data.uniform_(-initialization_range, initialization_range)

    def forward(self, x: tensor) -> tensor: 
        x: tensor = x.view(-1, self.config.sequence_length, self.config.num_assets * self.config.asset_dimension)
        x: tensor = self.positional_encoder(self.encoder(x) * math.sqrt(self.config.model_dimension))
        out: tensor = self.transformer_encoder(x) 
        return out 

class ForecastingNet(nn.Module): 
    """A forecasting network which embeds the time series 
    data and later aggregates it with static predictors 
    to produce a final forecast.
    """
    def __init__(self, config: ForecastingConfig): 
        super(ForecastingNet, self).__init__() 
        self.model_type: str = "ForecastingNet"
        config.embedding_input_dimension = config.num_assets * config.asset_dimension
        self.time_series_encoder: layer = EmbeddingNet(config)
        self.aggregation_net: layer = FeedForwardNet(config)

    def forward(self, time_series: tensor) -> tensor: 
        time_series_encoding: tensor = self.time_series_encoder(time_series)
        out: tensor = self.aggregation_net(time_series_encoding)
        return out 
