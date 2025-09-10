def stampa(epochs, train_losses_norm, val_losses_norm,
           train_r2_norm, val_r2_norm,
           train_losses_orig, val_losses_orig,
           train_r2_orig, val_r2_orig,
           val_preds_orig, val_labels_orig):

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import make_interp_spline

    # Funzione per lo smoothing
    def smooth_plot(epochs, data, label, color):
        xnew = np.linspace(min(epochs), max(epochs), 300)
        if len(epochs) > 3:
            spl = make_interp_spline(epochs, data, k=3)
            smooth_data = spl(xnew)
            plt.plot(xnew, smooth_data, label=label, color=color)
        else:
            plt.plot(epochs, data, label=label, color=color, linestyle='--')
     #Plot Loss (Scala Originale)
    plt.figure(figsize=(12, 6))
    smooth_plot(epochs, train_losses_orig, 'Training Loss (Original Scale MSE)', 'green')
    smooth_plot(epochs, val_losses_orig, 'Validation Loss (Original Scale MSE)', 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss (Original Scale)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Plot Loss (Scala Originale) con limite sull'asse y
    plt.figure(figsize=(12, 6))
    smooth_plot(epochs, train_losses_orig, 'Training Loss (Original Scale MSE)', 'green')
    smooth_plot(epochs, val_losses_orig, 'Validation Loss (Original Scale MSE)', 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss (Original Scale)')
    plt.ylim(0, 50000)  # Imposta il limite inferiore a 0 e quello superiore a 0.1
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    smooth_plot(epochs, train_losses_orig, 'Training Loss (Original Scale MSE)', 'green')
    smooth_plot(epochs, val_losses_orig, 'Validation Loss (Original Scale MSE)', 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss (Original Scale)')
    plt.ylim(0, 10000)  # Imposta il limite inferiore a 0 e quello superiore a 0.1
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    smooth_plot(epochs, train_losses_orig, 'Training Loss (Original Scale MSE)', 'green')
    smooth_plot(epochs, val_losses_orig, 'Validation Loss (Original Scale MSE)', 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss (Original Scale)')
    plt.ylim(0, 2500)  # Imposta il limite inferiore a 0 e quello superiore a 0.1
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot R2 (Scala Originale)
    plt.figure(figsize=(12, 6))
    smooth_plot(epochs, train_r2_orig, 'Training R2 (Original Scale)', 'green')
    smooth_plot(epochs, val_r2_orig, 'Validation R2 (Original Scale)', 'red')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.title('Training and Validation R2 Score (Original Scale)')
    plt.ylim([-1, 1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("\nUltime predizioni (normalizzate vs reali):")
    for pred, true in zip(val_preds_orig[-10:], val_labels_orig[-10:]):
      pred = np.array(pred)
      true = np.array(true)
      if pred.ndim == 0:
        print(f"Pred: {pred:.4f} \t Reale: {true:.4f}")
      else:
        pred_str = " - ".join([f"{p:.4f}" for p in pred])
        true_str = " - ".join([f"{t:.4f}" for t in true])
        print(f"Pred: {pred_str} \t Reale: {true_str}") # Plot Loss (Normalizzato) con limite sull'asse y
