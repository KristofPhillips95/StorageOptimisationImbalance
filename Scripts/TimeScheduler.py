# !/usr/bin/env python3
from datetime import datetime
import pandas as pd
import os


# print("script running")
def write_time():
    filename = os.path.join(os.getcwd(), "time_file.csv")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    row = pd.DataFrame(data=[current_time], columns=["Timestamp"])

    if os.path.isfile(filename):
        df_times = pd.read_csv(filename, index_col=0)
        df_times = df_times.append(row, ignore_index=True)
    else:
        df_times = row
    # print("Time to write to file")
    # print(filename)
    df_times.to_csv(filename)
