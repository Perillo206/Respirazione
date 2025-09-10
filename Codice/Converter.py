from init import *
def Converter(filepath,output_folder,output_folder2,lista,IsTime):#filepath:Percorso da cui prendere i file da convertire
  import pandas as pd                                             #output_folder(Cartella del test) Output_folder2(Cartell del train)
  import os                                                       #Lista è una lista inizialmente vuota che serve per tenere traccia del campionamento dei file
  import datetime                                                 #IsTime sè true se la label è il tempo false altrimenti(Serve nel caso si vogliono fare predizione sul power)
  import sklearn                                                    
  import warnings
  import random
  from TimeToMinutes import time_to_minutes
  from Riga import riga
  warnings.filterwarnings("ignore", category=FutureWarning)
  for file in os.listdir(filepath):
    print(file)
    if file.endswith('.xlsx') or file.endswith('.xls'):

      foglio1=pd.read_excel(os.path.join(filepath, file),0,skiprows=2,header=None).dropna()
      foglio2=pd.read_excel(os.path.join(filepath, file),1,skiprows=2,header=None).dropna()
      foglio3 = pd.read_excel(os.path.join(filepath, file),2,skiprows=2,header=None).dropna()
      foglio4 = pd.read_excel(os.path.join(filepath, file),3,skiprows=2,header=None)
      dati_combinati = pd.concat([foglio1, foglio2], axis=0, ignore_index=True)
      dati_combinati = pd.concat([dati_combinati, foglio3], axis=0, ignore_index=True)
      LabelRc=time_to_minutes(dati_combinati.iloc[len(dati_combinati)-1,0].strftime('%M:%S'))
      LabelAt=time_to_minutes(dati_combinati.iloc[len(dati_combinati)-2,0].strftime('%M:%S'))
      #print(dati_combinati)
      #print(riga(file,time_to_minutes(LabelRc)))
      #print(LabelRc)

      #print(foglio4)
      #print(foglio4.shape)
      #print(foglio1.shape)
      Secondi_Passati=0;
      for i in range(len(foglio1)):
        #print(type(foglio1.iloc[i, 0]))
        foglio1.iloc[i,0]=foglio1.iloc[i,0].strftime('%M:%S')
        foglio1.iloc[i,0]=time_to_minutes(foglio1.iloc[i,0])
        if ((i<len(foglio1))and (i!=0)):
          Secondi_Passati+=(foglio1.iloc[i,0])-(foglio1.iloc[i-1,0])
          #print(str(foglio1.iloc[i,0]) + "-" + str(foglio1.iloc[i-1,0])+"="+str(Secondi_Passati))
        if not (foglio4.empty):
          for j in range(1,len(foglio1.iloc[i])-1):
            foglio1.iloc[i,j]= foglio1.iloc[i,j]/foglio4.iloc[0,j]
      for i in range(len(foglio2)):
        foglio2.iloc[i,0]=foglio2.iloc[i,0].strftime('%H:%M:%S')
        foglio2.iloc[i,0]=time_to_minutes(foglio2.iloc[i,0])
        if not (foglio4.empty):
          for j in range(1,len(foglio2.iloc[i])-1):
            foglio2.iloc[i,j]= foglio2.iloc[i,j]/foglio4.iloc[0,j]

      for i in range(len(foglio3)):
        foglio3.iloc[i,0]=foglio3.iloc[i,0].strftime('%H:%M:%S')
        foglio3.iloc[i,0]=time_to_minutes(foglio3.iloc[i,0])
        if not (foglio4.empty):
          for j in range(1,len(foglio3.iloc[i])-1):
            foglio3.iloc[i,j]= foglio3.iloc[i,j]/foglio4.iloc[0,j]
      tempoMedio=Secondi_Passati/(len(foglio1)-1)#Serve per la suddivisione dei file in base a i secondi di campionamento
      #print(foglio1.iloc[:,0])
      Riga=riga(foglio1.iloc[:,0],LabelRc)# Serve per ottenere la riga della label Rc e poi fare il taglio
      R = random.randint(0, 25)
      #foglio1=foglio1.iloc[:Riga+1+R,:]
      #print(type(foglio1.iloc[5,0]))
      #print(type(LabelRc))
      lista[int(tempoMedio)]+=1
      if(IsTime):
        Min=foglio1.iloc[0,0]
        Max=foglio1.iloc[len(foglio1)-1,0]
      else:
        Min=min(foglio1.iloc[:,8])
        Max=max(foglio1.iloc[:,8])
      dati_combinati = pd.concat([foglio1, foglio2], axis=0, ignore_index=True)
      dati_combinati = pd.concat([dati_combinati, foglio3], axis=0, ignore_index=True)
      Colonne2=[0,1,2,3,4,5,6,7,8]
      Scaler=sklearn.preprocessing.MinMaxScaler()
      foglio1=pd.DataFrame(Scaler.fit_transform(foglio1[Colonne2]))
      dati_combinati=pd.DataFrame(Scaler.fit_transform(dati_combinati[Colonne2]))
      foglio2.iloc[0] = foglio2.iloc[0].astype(float)
      foglio2.iloc[0]=dati_combinati.iloc[len(dati_combinati)-2]
      foglio3.iloc[0] = foglio3.iloc[0].astype(float)
      foglio3.iloc[0]=dati_combinati.iloc[len(dati_combinati)-1]
      os.makedirs(output_folder, exist_ok=True)
      nome_file_output = os.path.splitext(file)[0] + "_scalato.xlsx"
      #percorso_output = os.path.join(output_folder, nome_file_output)
      percorso_output = os.path.join(output_folder, nome_file_output)
      #if (int(tempoMedio)==10):
        #percorso_output = os.path.join(folder10s, nome_file_output)
      #elif (int(tempoMedio)==15):
        #percorso_output = os.path.join(folder15s, nome_file_output)
      minmax_df = pd.DataFrame({'Tipo': ['Min', 'Max','Tempo di Campionamento Medio'],'Valore': [Min,Max,tempoMedio]})
      with pd.ExcelWriter(percorso_output, engine='xlsxwriter') as writer:

                foglio1.to_excel(writer, index=False, sheet_name='Foglio_Scalato1')
                foglio2.to_excel(writer, index=False, sheet_name='AT')
                foglio3.to_excel(writer, index=False, sheet_name='RC')
                minmax_df.to_excel(writer, index=False, sheet_name='MinMax')

    elif (os.path.isdir(os.path.join(filepath,file))):
      Converter(os.path.join(filepath,file),output_folder2,output_folder2,lista,IsTime)
       #Colonne=['Rf','VO2','VCO2','VE/VO2','VE/VCO2','HR','VO2/HR','Power']
      #if ('Power'not in foglio.columns):
        #Colonne=['Rf','VO2','VCO2','VE/VO2','VE/VCO2','HR','VO2/HR','Load']
