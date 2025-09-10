def riga(input,number):
  preRes=number
  res=number
  for i in range(len(input)):
    res=number-input.iloc[i]
    if (res>0):
      preRes=res
    if (res<0):
        if(abs(preRes)<abs(res)):
          return i-1
        else:
          return i
    if (res==0):
        return i
