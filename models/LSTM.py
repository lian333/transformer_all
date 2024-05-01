import torch
import torch.nn as nn
from layers.Embed import DataEmbedding

class Model(nn.Module):
    """
    LSTM model for sequence processing with configurable architecture.
    This model is designed to be flexible for various sequence processing tasks.
    Hyperparameters such as the number of layers, hidden dimension, and bidirectional
    flag should be adjusted based on the specific application.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding layer
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            bidirectional=False,
            batch_first=True
        )
        
        # Depending on bidirectional or not, adjust the output linear layer size
        num_directions = 1 # if bidirectional else 1
        self.projection = nn.Linear(configs.d_model * num_directions, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None):
        # Embedding for encoder
        x_enc = self.enc_embedding(x_enc, x_mark_enc)
        # Processing sequence through LSTM
        lstm_out, _ = self.lstm(x_enc)
        lstm_out = self.projection(lstm_out)

        if self.output_attention:
            return lstm_out[:, -self.pred_len:, :], None  # Attention not applicable in LSTM
        else:
            return lstm_out[:, -self.pred_len:, :]  # [B, L, D]
