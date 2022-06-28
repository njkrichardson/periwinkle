import dataclasses
import math 

import torch 
from torch import nn 

from type_aliases import * 

@dataclasses.dataclass
class EmbeddingConfig: 
    """Configuration dataclass for the time-series 
    embedding network. Contains attributes to encapsulate 
    the architectural parameters of the network. 
    """
    sequence_length: Optional[int] = 64 
    model_dimension: Optional[int] = 64 
    num_heads: Optional[int] = 4
    hidden_dimension: Optional[int] = 16 
    num_layers: Optional[int] = 4 
    dropout_rate: Optional[float] = 0.

@dataclasses.dataclass
class ForecastingConfig: 
    """Configuration dataclass for the end-to-end
    forecasting network.

    TODO add an aggegation config for the feedforward net
    """
    embedding_config: Optional[EmbeddingConfig] = dataclasses.field(default_factory=EmbeddingConfig)

class FeedForwardNet(nn.Module): 
    """Vanilla fully-connected multi-layer 
    perceptron (i.e., a feedforward neural net). 

    TODO: make layer sizes parametric 
    """
    def __init__(self, activation: str="relu"): 
        super(FeedForwardNet, self).__init__()
        nonlinearity: layer = nn.ReLU() if activation == "relu" else NotImplementedError
        self.ravel: layer = nn.Flatten()
        self.net: layer = nn.Sequential(
                nn.Linear(80, 128), nonlinearity, 
                nn.Linear(128, 64)
                )

    def forward(self, x: tensor) -> tensor: 
        x: tensor = self.ravel(tensor) 
        out: tensor = self.net(x) 
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
    def __init__(self, config: EmbeddingConfig): 
        super(EmbeddingNet, self).__init__() 
        self.encoder: layer = nn.Linear(config.sequence_length, config.model_dimension)
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
        x = self.positional_encoder(self.encoder(x) * math.sqrt(self.config.model_dimension))
        out: tensor = self.transformer_encoder(x) 
        return out 

class ForecastingNet(nn.Module): 
    """A forecasting network which embeds the time series 
    data and later aggregates it with static predictors 
    to produce a final forecast.
    """
    def __init__(self, config: ForecastingConfig): 
        super(ForecastingNet, self).__init__() 
        self.time_series_encoder: layer = EmbeddingNet(config.embedding_config)
        self.aggregation_net: layer = FeedForwardNet() 

    def forward(self, inputs: tuple) -> tensor: 
        time_series, auxiliary = inputs 
        time_series_encoding: tensor = self.time_series_encoder(time_series)
        out: tensor = self.aggregation_net(torch.cat(time_series_encoding, auxiliary))
        return out 
