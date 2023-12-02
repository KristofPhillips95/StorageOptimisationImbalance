import datetime
# import pandas as pd
# import os
import requests
from predict_imb_price import pred_SI

def write_item_API_and_reschedule(scheduler, interval=60,index=0):
    # Schedule the next call first
    index += 1
    scheduler.enter(interval, index, write_item_API_and_reschedule, (scheduler, interval,index,))
    # Then do your stuff
    write_item_API(index)


def write_item_API(index):
    print(index)
    SI_FC, last_si, quantiles = pred_SI(dev='cpu')

    prices_fc_spread = dict()
    last_si_time = last_si[1]
    fc_times = [(last_si_time + datetime.timedelta(minutes=15 * fc_step)).strftime("%H:%M:%S") for fc_step in range(8)]

    for i, fc_time in enumerate(fc_times):
        prices_fc_spread[fc_time] = [str(SI_FC[i, q]) for q in range(SI_FC.shape[1])]
    data = {
        "id": index,
        "time": last_si_time.strftime('%H:%M:%S'),
        "imba_price": last_si[0],
        "imba_price_fc": 1,
        "charge": 1,
        "soc": 1,
        "fc_spread": prices_fc_spread,
        "quantiles": quantiles
    }


    print(f"Writing now to API:", last_si_time)

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