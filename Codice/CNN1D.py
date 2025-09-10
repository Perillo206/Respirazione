import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.interpolate import make_interp_spline
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import matplotlib.pyplot as plt
# Definisci la tua classe TimeSeriesDataset (assicurati che restituisca sequenze e label come tensori)
#0.29910652740093757
class CNN1D(nn.Module):
    def __init__(self, input_channels, sequence_length, num_filters=16, kernel_size=3, output_dim=2, dropout_rate=  0.44192912814990604,):
        """
        Architettura di una CNN 1D con Dropout.

        Args:
            input_channels (int): Numero di feature nella serie temporale.
            sequence_length (int): La lunghezza fissa a cui hai paddato le tue sequenze.
            num_filters (int): Numero di filtri convoluzionali.
            kernel_size (int): Dimensione del kernel convoluzionale.
            output_dim (int): Dimensione dell'output (2 per prevedere due valori).
            dropout_rate (float): Probabilità di dropout (valore tra 0 e 1).
        """
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding='same')#Secondo layer
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(dropout_rate)  # Dropout dopo la convoluzione
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout_fc = nn.Dropout(dropout_rate)    # Dropout prima del layer fully connected
        self.fc = nn.Linear(num_filters * sequence_length, output_dim)

    def forward(self, x):
        """
        Forward pass della CNN 1D con Dropout.

        Args:
            x (torch.Tensor): Tensor di input con forma (batch_size, sequence_length, input_channels).

        Returns:
            torch.Tensor: Tensor di output con forma (batch_size, output_dim).
        """
        # Trasponi per avere (batch_size, input_channels, sequence_length) per la Conv1d
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout_conv(x)  # Applica il dropout dopo la ReLU del layer convoluzionale
        x = self.flatten(x)
        x = self.dropout_fc(x)    # Applica il dropout prima del layer fully connected
        x = self.fc(x)
        return x
