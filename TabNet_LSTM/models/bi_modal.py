import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LSTM_with_attention import AttnLSTM
from pytorch_tabnet.tab_network import TabNet
import numpy as np

import warnings
warnings.filterwarnings('ignore')

class Bi_Model(nn.Module):
    def __init__(self, seq_input_size, seq_hidden, seq_num_layers,
                 tab_input_size, tab_hidden, cat_idxs, cat_dims, h_merge, num_class):
        super(Bi_Model, self).__init__()
        self.att_lstm = AttnLSTM(
            input_size=seq_input_size,
            hidden_size=seq_hidden,
            num_layers=seq_num_layers)
        self.tabnet = TabNet(
            input_dim = tab_input_size,
            output_dim=tab_hidden,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims
        )
        self.merge_type=h_merge
        if self.merge_type == "concat":
            self.fc = nn.Linear(seq_hidden+tab_hidden, num_class)
        elif self.merge_type == "mean":
            self.fc = nn.Linear(seq_hidden, num_class)

    def forward(self, seq_inputs, tab_inputs):
        # LSTM with attention
        seq_output, weights = self.att_lstm(seq_inputs)

        # TabNet
        tab_output, _ = self.tabnet(tab_inputs)

        # Fc layer
        if self.merge_type == "concat":
            x = torch.cat((seq_output, tab_output), 1)
        elif self.merge_type == "mean":
            x = (seq_output+tab_output)/2
        output = self.fc(x)

        return output, weights


if __name__ == '__main__':

    cat_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    cat_dims = [3, 2, 3, 3, 3, 3, 3, 3, 5, 19, 6, 3, 1, 3, 3, 2, 2, 2, 6, 4, 5, 2]

    model = Bi_Model(seq_input_size=1,
                     seq_hidden=6,
                     seq_num_layers=1,
                     tab_input_size=22,
                     tab_hidden=6,
                     cat_idxs=cat_idxs,
                     cat_dims=cat_dims,
                     h_merge='mean',
                     num_class=9)

    seq_samples = torch.randn(12,10,1)
    tab_samples = torch.randint(22,(12,10))
    outputs, weights = model(seq_inputs = seq_samples, tab_inputs = tab_samples)