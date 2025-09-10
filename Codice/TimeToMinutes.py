def time_to_minutes(time_str):
    Tempo=time_str.split(':')
    if(len(Tempo)==2):
      return int(Tempo[0])*60+int(Tempo[1])
    else:
      return int(Tempo[0])*3600+int(Tempo[1])*60+int(Tempo[2])
