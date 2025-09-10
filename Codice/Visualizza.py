import pandas as pd
import matplotlib.pyplot as plt
import os
from TrainAndEvaluate import train_and_evaluate
from init import *
from TimeToMinutes import time_to_minutes
#Addestra il modello e stampa Mse,R2 e ogni file con relativi label e predizioni
#filepath è il percorso ai dati non normalizzati con già suddivisione in train e test 
def Visualizza(filepath,x,train_preds_orig,val_preds_orig,train_filenames,val_filenames):
  if(train_preds_orig==None):
    train_labels_orig, val_labels_orig, train_preds_orig, val_preds_orig, train_filenames, val_filenames =train_and_evaluate(Dati101Train, Dati101Test, 100)#Inserire le cartelle giuste per train e test
  Insieme=[]
  for file in os.listdir(filepath):
    if file.endswith('.xlsx') or file.endswith('.xls'):
# Carica il file Excel (sostituisci con il nome del tuo file)
      df = pd.read_excel(os.path.join(filepath, file),0,skiprows=2,header=None).dropna()
      df2 = pd.read_excel(os.path.join(filepath, file),1,skiprows=2,header=None).dropna()
      df3 = pd.read_excel(os.path.join(filepath, file),2,skiprows=2,header=None).dropna()
      TempoP1=0
      TempoP2=0
      for j in range(len(train_filenames)):
        file2=train_filenames[j]
        if (file[0:5]==file2[0:5]):
          print(file)
          print(file2)
          TempoP1=train_preds_orig[j][0]
          TempoP2=train_preds_orig[j][1]
          break
      for j in range(len(val_filenames)):
        file2=val_filenames[j]
        if (file[0:5]==file2[0:5]):
          print(file)
          print(file2)
          TempoP1=val_preds_orig[j][0]
          TempoP2=val_preds_orig[j][1]
          break

      for i in range(len(df3)):
        df3.iloc[i,0]=df3.iloc[i,0].strftime('%H:%M:%S')
        Tempo2=time_to_minutes(df3.iloc[i,0])# Tempo
        Tempo2=df3.iloc[i,8]#power
      for i in range(len(df2)):
        df2.iloc[i,0]=df2.iloc[i,0].strftime('%H:%M:%S')
        Tempo1=time_to_minutes(df2.iloc[i,0])
        Tempo1=df2.iloc[i,8]
# Converte la prima colonna (indice 0) in datetime, se contiene date
      for i in range(len(df)):
        df.iloc[i, 0] = df.iloc[i,0].strftime('%H:%M:%S')
        df.iloc[i,0]=time_to_minutes(df.iloc[i,0])
      if(Tempo1==Tempo2):
          Insieme.append(file)

# Crea il grafico con tutte le 6 serie temporali
      plt.figure(figsize=(14, 6))
# Cicla sulle colonne da 1 a 6 (esclude la colonna tempo)
      for i in range(1, 7):
          if (i==2):
            plt.plot(df.iloc[:, 0], df.iloc[:, i], label=f'V02 ')
          if (i==3):
            plt.plot(df.iloc[:, 0], df.iloc[:, i], label=f' VCO2 ')
# Etichette e layout
      plt.axvline(x=Tempo1, color='r', linestyle='--', label=f'Tempo1  ({Tempo1:.2f} min)')
      plt.axvline(x=Tempo2, color='b', linestyle='--', label=f'Tempo2 ({Tempo2:.2f} min)')
      plt.axvline(x=TempoP1, color='g', linestyle='--', label=f'Predizione1 ({TempoP1:.2f} min)')
      plt.axvline(x=TempoP2, color='y', linestyle='--', label=f'Predizione2 ({TempoP2:.2f} min)')
      plt.xlabel('Tempo')
      plt.ylabel('Valore')
      plt.title('Serie Temporali da Excel (2 colonne) di '+str(file))
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.show()
#Secondo grafico
      plt.figure(figsize=(14, 6))
      for i in range(1, 7):
          if (i==1):
            plt.plot(df.iloc[:, 0], df.iloc[:, i], label=f'Rf ')
          if (i==4):
            plt.plot(df.iloc[:, 0], df.iloc[:, i], label=f' VE/VO2 ')
          if (i==5):
            plt.plot(df.iloc[:, 0], df.iloc[:, i], label=f' VE/VCO2 ')
          if (i==6):
            plt.plot(df.iloc[:, 0], df.iloc[:, i], label=f' HR ')
          if(i==7):
            plt.plot(df.iloc[:, 0], df.iloc[:, i], label=f'VO2/HR ')
      plt.axvline(x=Tempo1, color='r', linestyle='--', label=f'Tempo1 ({Tempo1:.2f} min)')
      plt.axvline(x=Tempo2, color='b', linestyle='--', label=f'Tempo2 ({Tempo2:.2f} min)')
      plt.axvline(x=TempoP1, color='g', linestyle='--', label=f'Predizione1 ({TempoP1:.2f} min)')
      plt.axvline(x=TempoP2, color='y', linestyle='--', label=f'Predizione2 ({TempoP2:.2f} min)')
      plt.xlabel('Tempo')
      plt.ylabel('Valore')
      plt.title('Serie Temporali da Excel (5 colonne) di '+str(file))
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.show()
      x=x+1
      print(Insieme)
    elif (os.path.isdir(os.path.join(filepath,file))):
      Visualizza(os.path.join(filepath,file),10,train_preds_orig,val_preds_orig,train_filenames,val_filenames)
