from datetime import datetime
# import pandas as pd
# import os
import requests
from predict_imb_price import pred_SI

def write_item_API_and_reschedule(scheduler, interval=60,index=0):
    # Schedule the next call first
    index += 1
    scheduler.enter(interval, 1, write_item_API_and_reschedule, (scheduler, interval,index,))
    # Then do your stuff
    write_item_API(index)


def write_item_API(index):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    imba_price_fc = pred_SI()
    data = {
        "id": index,
        "time": current_time,
        # "imba_price": imba_price,
        "imba_price_fc":imba_price_fc,
        "charge": charge,
        "soc": soc
    }

    print(f"Writing now to API:", current_time)

    response = requests.put("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items", json=data)
    print(response.text)


    # filename = os.path.join("..", "time_file.csv")
    # row = pd.DataFrame(data=[current_time], columns=["Timestamp"])
    #
    # if os.path.isfile(filename):
    #     df_times = pd.read_csv(filename, index_col=0)
    #     df_times = pd.concat([df_times, row], ignore_index=True)
    # else:
    #     df_times = row
    # df_times.to_csv(filename)