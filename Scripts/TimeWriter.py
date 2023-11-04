from datetime import datetime
import pandas as pd
import os


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

def write_time_and_reschedule(scheduler, interval=60):
    # Schedule the next call first
    scheduler.enter(interval, 1, write_time_and_reschedule, (scheduler, interval,))
    # Then do your stuff
    write_time()