from datetime import datetime
import pandas as pd
import os

filename = os.path.join("..","time_file.csv")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
row = pd.DataFrame(data=[current_time], columns=["Timestamp"])

if os.path.isfile(filename):
    df_times = pd.read_csv(filename,index_col=0)
    df_times = df_times.append(row,ignore_index=True)
else:
    df_times = row
df_times.to_csv(filename)
