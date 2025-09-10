import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.metrics import r2_score
import os
from Collate import collate3_fn
from CNN1D import CNN1D
from Stampa  import stampa
from TimeSeriesDataset import TimeSeriesDataset
from init import *
def train_and_evaluate(output_folder_train, output_folder_val, num_epochs):#Addestra e stampa solamente i grafici relaviti a Mse e R2
    # Inizializzazioni
    torch.manual_seed(42)
    train_losses_norm, val_losses_norm = [], []
    train_r2_norm, val_r2_norm = [], []
    train_losses_orig, val_losses_orig = [], []
    train_r2_orig, val_r2_orig = [], []
    train_inputs_orig, val_inputs_orig = [], []
    train_last_pred, val_last_pred = [], []
    best_val_loss = float("inf")

    train_dataset = TimeSeriesDataset(output_folder_train)
    validation_dataset = TimeSeriesDataset(output_folder_val)
  
    # LISTE PER I NOMI DEI FILE
    train_original_filenames = []
    val_original_filenames = []

    input_channels = train_dataset[0][0].shape[1]
    if (train_dataset._max_len() > validation_dataset._max_len()):
        max_len = train_dataset._max_len()
    else:
        max_len = validation_dataset._max_len()
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, collate_fn=lambda b: collate3_fn(b, max_len))
    validation_loader = DataLoader(validation_dataset, batch_size=20, shuffle=False, collate_fn=lambda b: collate3_fn(b, max_len))
    model = CNN1D(input_channels=input_channels,sequence_length=max_len)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    criterion = nn.MSELoss()
    #optimizer=optim.SGD(model.parameters(), lr=0.00029057058787255557)
    optimizer = optim.Adam(model.parameters(), lr= 0.000778356361960688)

    # InsiemeRighe e InsiemeRichePred sembrano non usati o usati impropriamente,
    # li lascio ma verifica il loro scopo.
    InsiemeRighe=[]
    InsiemeRichePred=[]

    # Crea cartella per i checkpoint
    checkpoint_dir = os.path.join(filepath2, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_epoch_loss_norm = 0
        train_epoch_loss_orig = 0
        train_labels_norm, train_preds_norm = [], []
        train_labels_orig, train_preds_orig = [], []

        # CICLO DI TRAINING
        for inputs, labels, min_vals, max_vals, filenames in train_loader: # 'filenames' sono i nomi dei file del training batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if labels.dim() > 2:
                labels = labels.squeeze(1)

            loss_norm = criterion(outputs, labels)
            optimizer.zero_grad()
            loss_norm.backward()
            optimizer.step()
            train_epoch_loss_norm += loss_norm.item()
            train_labels_norm.extend(labels.cpu().numpy().tolist())
            train_preds_norm.extend(outputs.detach().cpu().numpy().tolist())

            batch_loss_orig = 0
            batch_labels_orig, batch_preds_orig = [], []
            # NON INIZIALIZZARE batch_labels_orig2, batch_preds_orig2 QUI
            # Se le usi, dovrebbero essere inizializzate PRIMA del ciclo for batch,
            # oppure il loro scopo è solo locale a questo ciclo 'for i'

            for i in range(labels.shape[0]):
                min_val = min_vals[i].item()
                max_val = max_vals[i].item()
                scale = max_val - min_val
                inp = inputs[i].cpu().numpy()
                label_orig = labels[i].cpu().numpy() * scale + min_val
                pred_orig = outputs[i].detach().cpu().numpy() * scale + min_val
                batch_loss_orig += np.mean((label_orig - pred_orig) ** 2)
                train_inputs_orig.append(inp)
                batch_labels_orig.append(label_orig)
                batch_preds_orig.append(pred_orig)

                if (epoch == num_epochs - 1):
                    train_original_filenames.append(filenames[i]) 

            train_epoch_loss_orig += batch_loss_orig / labels.shape[0]
            train_labels_orig.extend(batch_labels_orig)
            train_preds_orig.extend(batch_preds_orig)


            if(epoch == num_epochs - 1):
                train_last_pred.extend(batch_preds_orig) # Usa batch_preds_orig o un'altra lista apposita


        train_losses_norm.append(train_epoch_loss_norm / len(train_loader))
        train_losses_orig.append(train_epoch_loss_orig / len(train_loader))
        train_r2_norm.append(r2_score(train_labels_norm, train_preds_norm) if train_labels_norm else np.nan)
        train_r2_orig.append(r2_score(train_labels_orig, train_preds_orig) if train_labels_orig else np.nan)

        # CICLO DI VALIDAZIONE
        model.eval()
        val_epoch_loss_norm = 0
        val_epoch_loss_orig = 0
        val_labels_norm, val_preds_norm = [], []
        val_labels_orig, val_preds_orig = [], []

        # Variabile per raccogliere le predizioni finali del batch di validazione,
        # se val_last_pred deve contenere solo predizioni dell'ultimo epoch
        temp_val_preds_for_last_epoch = []

        with torch.no_grad():
            for val_inputs, val_labels, val_mins, val_maxs, val_filenames in validation_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                if val_labels.dim() > 2:
                    val_labels = val_labels.squeeze(1)

                val_loss_norm = criterion(val_outputs, val_labels)
                val_epoch_loss_norm += val_loss_norm.item()
                val_labels_norm.extend(val_labels.cpu().numpy().tolist())
                val_preds_norm.extend(val_outputs.detach().cpu().numpy().tolist())

                batch_loss_orig = 0
                batch_labels_orig, batch_preds_orig = [], []

                for i in range(val_labels.shape[0]):
                    min_val = val_mins[i].item()
                    max_val = val_maxs[i].item()
                    inp = val_inputs[i].cpu().numpy()
                    scale = max_val - min_val
                    label_orig = val_labels[i].cpu().numpy() * scale + min_val
                    pred_orig = val_outputs[i].detach().cpu().numpy() * scale + min_val
                    batch_loss_orig += np.mean((label_orig - pred_orig) ** 2)
                    batch_labels_orig.append(label_orig)
                    batch_preds_orig.append(pred_orig)
                    val_inputs_orig.append(inp) # Se vuoi tutti gli input di validazione

                    # Questa riga era già corretta, ma ora il contesto è chiaro
                    if (epoch == num_epochs - 1):
                        val_original_filenames.append(val_filenames[i])
                        temp_val_preds_for_last_epoch.append(pred_orig) # Raccogli qui per val_last_pred
                    # if(epoch==num_epochs-1):
                    # print(val_labels[i,0].numpy())
                    # print(val_labels[i,1].numpy())
                    # print(val_labels[i,0].numpy())
                    # print(PrimaColonna[i][0])
                    # print(val_preds_norm[i][0])
                    # InsiemeRighe.append(riga(PrimaColonna[i],val_labels[i,0].numpy()))
                    # InsiemeRighe.append(riga(PrimaColonna[i],val_labels[i,1].numpy()))

                val_epoch_loss_orig += batch_loss_orig / val_labels.shape[0]
                val_labels_orig.extend(batch_labels_orig)
                val_preds_orig.extend(batch_preds_orig)

            # Fuori dal ciclo 'for i', ma ancora dentro il ciclo 'for batch'
            if(epoch == num_epochs - 1):
                # Estendi val_last_pred con i valori raccolti da questo batch
                val_last_pred.extend(temp_val_preds_for_last_epoch)


        current_val_loss = val_epoch_loss_norm / len(validation_loader)
        val_losses_norm.append(current_val_loss)
        val_losses_orig.append(val_epoch_loss_orig / len(validation_loader))
        val_r2_norm.append(r2_score(val_labels_norm, val_preds_norm) if val_labels_norm else np.nan)
        val_r2_orig.append(r2_score(val_labels_orig, val_preds_orig) if val_labels_orig else np.nan)

    epochs_np = np.array(range(1, num_epochs + 1))
    xnew = np.linspace(epochs_np.min(), epochs_np.max(), 300)
    stampa(epochs_np, train_losses_norm, val_losses_norm, train_r2_norm, val_r2_norm,
           train_losses_orig, val_losses_orig, train_r2_orig, val_r2_orig,
           val_preds_orig, val_labels_orig)

    return (
        train_labels_orig,
        val_labels_orig,
        train_last_pred,
        val_last_pred,
        train_original_filenames, # Nomi dei file di training (popolati nell'ultimo epoch)
        val_original_filenames    # Nomi dei file di validazione (popolati nell'ultimo epoch)
    )
