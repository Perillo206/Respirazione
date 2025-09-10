import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Collate import collate3_fn
from CNN1D import CNN1D
from Stampa  import stampa
from TimeSeriesDataset import TimeSeriesDataset
from init import *
import numpy as np

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    num_filters = trial.suggest_categorical("num_filters", [16, 32, 64])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    # Dati da modificare manualmente 
    train_dataset = TimeSeriesDataset(Dati101Train)
    validation_dataset = TimeSeriesDataset(Dati101Test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda b: collate3_fn(b, max_len))
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=lambda b: collate3_fn(b, max_len))


    input_channels = train_dataset[0][0].shape[1]
    sequence_length = max_len
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Modello
    model = CNN1D(input_channels=input_channels,
                  sequence_length=sequence_length,
                  num_filters=num_filters,
                  kernel_size=kernel_size,
                  dropout_rate=dropout_rate).to(device)

    # Ottimizzatore
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()
    num_epochs = 50
    patience = 1000
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Training con early stopping
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels, _, _ ,_ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if labels.dim() > 2:
                labels = labels.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validazione dopo ogni epoca
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, _, _ ,_ in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if labels.dim() > 2:
                    labels = labels.squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(validation_loader)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_val_loss
