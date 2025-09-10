import optuna
from HypConfigAux import objective
from init import *
from TrainAndEvaluate import train_and_evaluate
def HypConfig():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=65)
    print("Numero di trial eseguiti:", len(study.trials))
    print("Migliori parametri:", study.best_params)
    print("Miglior valore obiettivo (loss di validazione):", study.best_value)


    try:
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()
        optuna.visualization.plot_parallel_coordinate(study).show()
    except ImportError:
        print("Per visualizzare i risultati, installa 'plotly'.")
