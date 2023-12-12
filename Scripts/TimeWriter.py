from datetime import datetime
import pandas as pd
import os
import requests


def write_time():
    filename = os.path.join("..", "time_file.csv")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    row = pd.DataFrame(data=[current_time], columns=["Timestamp"])

    if os.path.isfile(filename):
        df_times = pd.read_csv(filename, index_col=0)
        df_times = pd.concat([df_times, row], ignore_index=True)
    else:
        df_times = row

    print(f"Writing now to: {filename}", df_times)
    df_times.to_csv(filename)

def write_time_API_and_reschedule(scheduler, interval=60,index=0):
    # Schedule the next call first
    scheduler.enter(interval, 1, write_time_API_and_reschedule, (scheduler, interval,index,))
    # Then do your stuff
    write_time_API(index)

def write_time_API(index):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    data = {
    "id": str(index),
    "price": 2,
    "name": str(current_time)
        }
    data = {
        "id": "1",
        "price": 0,
        "name": str(current_time)
    }

    print(f"Writing now to API:", current_time)
    response = requests.put('https://d5mvov6803.execute-api.eu-north-1.amazonaws.com/items', json=data)
    print(f"Respones API:", response.text)


    filename = os.path.join("..", "time_file.csv")
    row = pd.DataFrame(data=[current_time], columns=["Timestamp"])

    if os.path.isfile(filename):
        df_times = pd.read_csv(filename, index_col=0)
        df_times = pd.concat([df_times, row], ignore_index=True)
    else:
        df_times = row
    df_times.to_csv(filename)