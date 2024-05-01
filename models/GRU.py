import torch
import torch.nn as nn
from layers.Embed import DataEmbedding

class Model(nn.Module):
    """
    GRU model for sequence processing with configurable architecture.
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
        # GRU Layer
        self.gru = nn.GRU(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            bidirectional=False,
            batch_first=True
        )
        
        self.projection = nn.Linear(configs.d_model, configs.c_out)  # Output projection layer

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None):
        # Embedding for encoder input
        x_enc = self.enc_embedding(x_enc, x_mark_enc)
        # Processing sequence through GRU
        gru_out, _ = self.gru(x_enc)
        gru_out = self.projection(gru_out)

        if self.output_attention:
            return gru_out[:, -self.pred_len:, :], None  # Attention not applicable in GRU
        else:
            return gru_out[:, -self.pred_len:, :]  # [B, L, D]
