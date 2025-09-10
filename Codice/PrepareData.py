def prepareData(output_folder,i):
  #Mi restituisce un dizionario con chiave i percorsi e valore un insieme in cui nel primo elemento è rinchiuso il dataframe del test mentre nel secondo l'AT e nel terzo RC
  import pandas as pd
  import os
  import datetime
  import sklearn
  import numpy as np
  from init import Test, AT, RC,MinMax,max_len
  out_data=dict()
  max_len=0
  for file in os.listdir(output_folder):
    if file.endswith('.xlsx') or file.endswith('.xls'):
      foglio1=pd.ExcelFile(os.path.join(output_folder, file))
      test_df = pd.read_excel(foglio1, Test)
      leng=len(test_df)
      if(leng>max_len):
        max_len=leng
      at_df = pd.read_excel(foglio1, 'AT')
      rc_df = pd.read_excel(foglio1, 'RC')
      minmax_df = pd.read_excel(foglio1, 'MinMax')
      df_list = [test_df, at_df, rc_df,minmax_df]
      out_data[file]=df_list[i]
      #File=out_data[Percorso_Completo]
      #print(File['Foglio1_Scalato'])
  return out_data,max_len
